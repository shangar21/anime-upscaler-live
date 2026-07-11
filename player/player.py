#!/usr/bin/env python3
"""
DeepStream upscaling player with APISR TensorRT backend.

Pipeline (when upscaler is active):
  filesrc -> parsebin -> nvv4l2decoder -> nvstreammux
  -> nvvideoconvert -> [caps filter: NVMM RGBA]
  -> nvdsvideotemplate (APISR TRT upscaler)
  -> nvvideoconvert -> nv3dsink

The caps filter between the two nvvideoconverts is required -- without it,
nvvideoconvert drops the memory:NVMM caps feature and the plugin receives
plain system-memory buffers instead of NvBufSurfaces, causing a crash.
Confirmed necessary during testing with gst-launch -v caps tracing.

Audio plays via decodebin off the audio pad -> autoaudiosink.
Subtitles/other tracks are drained to a fakesink.
"""
import os

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gst, GstPbutils, Gtk, GLib

Gst.init(None)

# Path to the compiled nvdsvideotemplate custom library.
# Set to None to run in decode-only mode (no upscaling).
CUSTOMLIB_PATH = "/opt/nvidia/deepstream/deepstream-9.0/lib/libcustom_videoimpl.so"

# Absolute path to the TensorRT engine. Must match the version DeepStream's
# TensorRT runtime (10.16) expects -- i.e. built with the pinned trtexec,
# not the 11.1 one that apt may have on PATH by default.
ENGINE_PATH = "/home/shangar21/Documents/upscaler/model/trt_engines/apisr_fp16.trt"

# HR-pixel overlap between adjacent tiles. Default (16) matches quant.py's
# bench default. Reduce to 8 for fewer tiles/frame and better GPU utilization
# if seams aren't visible; increase if you see seam artifacts.
OVERLAP_HR = 16

# Set True to drop all audio (diagnostic: isolates whether the audio clock is
# causing the pipeline stall vs. a video-path issue). If video plays through
# with this True, the stall is audio-clock-related.
DISABLE_AUDIO = False 

# Set True to replace nv3dsink with fakesink (drops frames, no window).
# If frames keep processing past 4 with this on, the stall is nv3dsink-specific
# and we need to switch to a different display sink.
USE_FAKESINK = False 


class PlayerWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="DeepStream Upscale Player")
        self.set_default_size(420, 160)
        self.pipeline = None
        self.mux = None
        self.duration_ns = 0
        self.user_seeking = False
        self.position_timeout_id = None
        self._audio_connected = False

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(20)
        box.set_margin_bottom(20)
        box.set_margin_start(20)
        box.set_margin_end(20)
        self.add(box)

        self.status_label = Gtk.Label(label="No file loaded")
        box.add(self.status_label)

        self.seek_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 100, 1)
        self.seek_scale.set_draw_value(False)
        self.seek_scale.set_sensitive(False)
        self.seek_scale.connect("button-press-event", self.on_seek_start)
        self.seek_scale.connect("button-release-event", self.on_seek_end)
        box.add(self.seek_scale)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        box.add(button_box)

        open_button = Gtk.Button(label="Open Video…")
        open_button.connect("clicked", self.on_open_clicked)
        button_box.add(open_button)

        self.stop_button = Gtk.Button(label="Stop")
        self.stop_button.connect("clicked", self.on_stop_clicked)
        self.stop_button.set_sensitive(False)
        button_box.add(self.stop_button)

        self.connect("destroy", self.on_destroy)

    # ---------- helpers ----------

    def _upscaler_active(self):
        return (
            CUSTOMLIB_PATH
            and os.path.exists(CUSTOMLIB_PATH)
            and ENGINE_PATH
            and os.path.exists(ENGINE_PATH)
        )

    # ---------- file picking ----------

    def on_open_clicked(self, widget):
        dialog = Gtk.FileChooserDialog(title="Select a video file", parent=self,
                                        action=Gtk.FileChooserAction.OPEN)
        dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL,
                            "_Open", Gtk.ResponseType.OK)
        video_filter = Gtk.FileFilter()
        video_filter.set_name("Video files")
        for pattern in ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"):
            video_filter.add_pattern(pattern)
        dialog.add_filter(video_filter)

        response = dialog.run()
        filepath = dialog.get_filename() if response == Gtk.ResponseType.OK else None
        dialog.destroy()

        if filepath:
            self.start_playback(filepath)

    # ---------- pipeline construction ----------

    def start_playback(self, filepath):
        self.stop_pipeline()
        self._audio_connected = False   # reset so the new file's first audio track connects

        self.status_label.set_text(f"Probing {os.path.basename(filepath)}…")
        try:
            width, height, duration_ns = self._probe_video_info(filepath)
        except Exception as exc:
            self.status_label.set_text(f"Failed to read file: {exc}")
            return
        self.duration_ns = duration_ns

        self.pipeline = Gst.Pipeline.new("player")

        src = Gst.ElementFactory.make("filesrc", "src")
        src.set_property("location", filepath)

        parsebin = Gst.ElementFactory.make("parsebin", "parsebin")

        self.mux = Gst.ElementFactory.make("nvstreammux", "mux")
        self.mux.set_property("batch-size", 1)
        self.mux.set_property("width", width)
        self.mux.set_property("height", height)

        # Build the fixed chain downstream of nvstreammux.
        # With upscaler active:
        #   mux -> preconv -> [NVMM RGBA caps filter] -> upscaler -> postconv -> sink
        # Without:
        #   mux -> postconv -> sink
        fixed_chain = [self.mux]

        if self._upscaler_active():
            # nvvideoconvert before the upscaler -- converts P010/NV12 etc. to RGBA.
            preconv = Gst.ElementFactory.make("nvvideoconvert", "preconv")

            # Caps filter: forces memory:NVMM + RGBA on the link between preconv
            # and nvdsvideotemplate. Without this, nvvideoconvert silently drops
            # the NVMM caps feature and the plugin receives system-memory buffers
            # instead of NvBufSurfaces, causing an immediate SIGSEGV.
            capsfilter = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
            caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
            capsfilter.set_property("caps", caps)

            upscaler = Gst.ElementFactory.make("nvdsvideotemplate", "upscaler")
            upscaler.set_property("customlib-name", CUSTOMLIB_PATH)
            # Pass engine path and overlap as separate customlib-props calls.
            # nvdsvideotemplate calls SetProperty once per customlib-props value,
            # so we set the property twice with different key=value strings.
            upscaler.set_property("customlib-props", f"engine-path:{ENGINE_PATH}")
            upscaler.set_property("customlib-props", f"overlap:{OVERLAP_HR}")

            fixed_chain += [preconv, capsfilter, upscaler]

        postconv = Gst.ElementFactory.make("nvvideoconvert", "postconv")
        if USE_FAKESINK:
            sink = Gst.ElementFactory.make("fakesink", "sink")
            sink.set_property("sync", False)
        else:
            sink = Gst.ElementFactory.make("nv3dsink", "sink")
            # Display frames as they arrive rather than blocking on each frame's
            # presentation timestamp. Prevents the pipeline from deadlocking when
            # the upscaler holds buffers and the sink waits on the clock, which
            # also starves the audio branch (pulse underflow).
            sink.set_property("sync", False)
        fixed_chain += [postconv, sink]

        for el in [src, parsebin] + fixed_chain:
            if el is None:
                self.status_label.set_text("Failed to create a required GStreamer element")
                return
            self.pipeline.add(el)

        src.link(parsebin)
        upstream = fixed_chain[0]
        for el in fixed_chain[1:]:
            if not upstream.link(el):
                print(f"Warning: could not link {upstream.get_name()} -> {el.get_name()}")
            upstream = el

        parsebin.connect("pad-added", self.on_parsebin_pad_added)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

        self.pipeline.set_state(Gst.State.PLAYING)

        mode = "4x upscaling" if self._upscaler_active() else "decode-only"
        self.status_label.set_text(
            f"Playing {os.path.basename(filepath)} ({width}x{height}, {mode})"
        )
        self.stop_button.set_sensitive(True)
        self.seek_scale.set_sensitive(True)
        self.seek_scale.set_range(0, max(duration_ns, 1))
        self.position_timeout_id = GLib.timeout_add(500, self.update_position)

    def _probe_video_info(self, filepath, timeout_seconds=5):
        discoverer = GstPbutils.Discoverer.new(timeout_seconds * Gst.SECOND)
        info = discoverer.discover_uri(Gst.filename_to_uri(filepath))
        streams = info.get_video_streams()
        if not streams:
            raise RuntimeError("no video stream found in this file")
        vstream = streams[0]
        return vstream.get_width(), vstream.get_height(), info.get_duration()

    def on_parsebin_pad_added(self, parsebin, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        struct = caps.get_structure(0)
        name = struct.get_name()

        if name.startswith("video/"):
            queue   = Gst.ElementFactory.make("queue", None)
            decoder = Gst.ElementFactory.make("nvv4l2decoder", None)
            self.pipeline.add(queue)
            self.pipeline.add(decoder)
            queue.sync_state_with_parent()
            decoder.sync_state_with_parent()
            pad.link(queue.get_static_pad("sink"))
            queue.link(decoder)
            mux_sink_pad = self.mux.get_request_pad("sink_0")
            decoder.get_static_pad("src").link(mux_sink_pad)

        elif name.startswith("audio/"):
            # Only route the first audio track for playback. Additional tracks
            # (dual-audio files) go straight to fakesink -- routing them all to
            # real sinks causes competing audio clocks and pipeline stall.
            if DISABLE_AUDIO or self._audio_connected:
                fakesink = Gst.ElementFactory.make("fakesink", None)
                fakesink.set_property("sync", False)
                self.pipeline.add(fakesink)
                fakesink.sync_state_with_parent()
                pad.link(fakesink.get_static_pad("sink"))
                return
            self._audio_connected = True

            queue           = Gst.ElementFactory.make("queue", None)
            audio_decodebin = Gst.ElementFactory.make("decodebin", None)
            self.pipeline.add(queue)
            self.pipeline.add(audio_decodebin)
            queue.sync_state_with_parent()
            audio_decodebin.sync_state_with_parent()
            pad.link(queue.get_static_pad("sink"))
            queue.link(audio_decodebin)
            audio_decodebin.connect("pad-added", self.on_audio_decodebin_pad_added)

        else:
            # subtitles / other tracks: drain to fakesink so the demuxer
            # doesn't stall waiting for a consumer.
            fakesink = Gst.ElementFactory.make("fakesink", None)
            fakesink.set_property("sync", False)
            self.pipeline.add(fakesink)
            fakesink.sync_state_with_parent()
            pad.link(fakesink.get_static_pad("sink"))

    def on_audio_decodebin_pad_added(self, decodebin, pad):
        # Gating to a single audio track happens in on_parsebin_pad_added,
        # so this only fires for the one track we chose to play.
        convert  = Gst.ElementFactory.make("audioconvert", None)
        resample = Gst.ElementFactory.make("audioresample", None)
        sink     = Gst.ElementFactory.make("autoaudiosink", None)
        for el in (convert, resample, sink):
            self.pipeline.add(el)
            el.sync_state_with_parent()
        convert.link(resample)
        resample.link(sink)
        pad.link(convert.get_static_pad("sink"))

    # ---------- seeking ----------

    def on_seek_start(self, widget, event):
        self.user_seeking = True

    def on_seek_end(self, widget, event):
        self.user_seeking = False
        if self.pipeline is not None:
            position_ns = int(self.seek_scale.get_value())
            self.pipeline.seek_simple(
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                position_ns,
            )

    def update_position(self):
        if self.pipeline is None:
            self.position_timeout_id = None
            return False
        if not self.user_seeking:
            ok, position_ns = self.pipeline.query_position(Gst.Format.TIME)
            if ok:
                self.seek_scale.set_value(position_ns)
        return True

    # ---------- lifecycle ----------

    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.status_label.set_text("Playback finished")
            self.stop_pipeline()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.status_label.set_text(f"Error: {err.message}")
            print("GStreamer debug info:", debug)
            self.stop_pipeline()
        return True

    def stop_pipeline(self):
        if self.position_timeout_id is not None:
            GLib.source_remove(self.position_timeout_id)
            self.position_timeout_id = None
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
        self.mux = None
        self.stop_button.set_sensitive(False)
        self.seek_scale.set_sensitive(False)

    def on_stop_clicked(self, widget):
        self.stop_pipeline()
        self.status_label.set_text("Stopped")

    def on_destroy(self, widget):
        self.stop_pipeline()
        Gtk.main_quit()


def main():
    win = PlayerWindow()
    win.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DeepStream dual-window player with APISR TensorRT backend.

Pipeline:
  filesrc -> parsebin -> nvv4l2decoder -> nvstreammux -> tee
    tee src_0 -> queue_up -> [preconv -> NVMM RGBA caps -> upscaler] -> postconv_up -> nv3dsink (Upscaled)
    tee src_1 -> queue_orig -> postconv_orig -> nv3dsink (Original)

Audio plays via decodebin off the audio pad -> autoaudiosink.
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

# Absolute path to the TensorRT engine. 
ENGINE_PATH = "/home/shangar21/Documents/upscaler/model/trt_engines/apisr_fp16.trt"

# HR-pixel overlap between adjacent tiles. 
OVERLAP_HR = 0 

DISABLE_AUDIO = False 
USE_FAKESINK = False 


class PlayerWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="DeepStream Dual-Window Player")
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
        self._audio_connected = False

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

        # Splitter and Queues for the two branches
        tee = Gst.ElementFactory.make("tee", "tee")
        queue_up = Gst.ElementFactory.make("queue", "queue_up")
        queue_orig = Gst.ElementFactory.make("queue", "queue_orig")

        # Set up elements for upscaler branch
        upscaler_elements = []
        if self._upscaler_active():
            preconv = Gst.ElementFactory.make("nvvideoconvert", "preconv")
            capsfilter = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
            caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
            capsfilter.set_property("caps", caps)

            upscaler = Gst.ElementFactory.make("nvdsvideotemplate", "upscaler")
            upscaler.set_property("customlib-name", CUSTOMLIB_PATH)
            upscaler.set_property("customlib-props", f"engine-path:{ENGINE_PATH}")
            upscaler.set_property("customlib-props", f"overlap:{OVERLAP_HR}")

            upscaler_elements = [preconv, capsfilter, upscaler]

        postconv_up = Gst.ElementFactory.make("nvvideoconvert", "postconv_up")
        #postconv_orig = Gst.ElementFactory.make("nvvideoconvert", "postconv_orig")
        # 1. Setup the converter and force Cubic interpolation (Algo-1 = Cubic on GPU)
        postconv_orig = Gst.ElementFactory.make("nvvideoconvert", "postconv_orig")
        postconv_orig.set_property("interpolation-method", 2) 

        # 2. Add a caps filter to explicitly set the output resolution.
        # Change scale_factor to match your APISR model (your C++ code implied 4x, but set to 2 if it's 2x).
        scale_factor = 4 
        capsfilter_orig = Gst.ElementFactory.make("capsfilter", "capsfilter_orig")
        caps_string = f"video/x-raw(memory:NVMM),width={width * scale_factor},height={height * scale_factor}"
        caps_orig = Gst.Caps.from_string(caps_string)
        capsfilter_orig.set_property("caps", caps_orig)

        # Output sinks
        if USE_FAKESINK:
            sink_up = Gst.ElementFactory.make("fakesink", "sink_up")
            sink_orig = Gst.ElementFactory.make("fakesink", "sink_orig")
        else:
            sink_up = Gst.ElementFactory.make("nv3dsink", "sink_up")
            sink_orig = Gst.ElementFactory.make("nv3dsink", "sink_orig")

        for sink in (sink_up, sink_orig):
            sink.set_property("sync", False)

        # Add everything to the pipeline
        #all_elements = [
        #    src, parsebin, self.mux, tee, 
        #    queue_up, queue_orig, postconv_up, postconv_orig, 
        #    sink_up, sink_orig
        #] + upscaler_elements

        all_elements = [
            src, parsebin, self.mux, tee, 
            queue_up, queue_orig, postconv_up, postconv_orig, capsfilter_orig, 
            sink_up, sink_orig
        ] + upscaler_elements

        for el in all_elements:
            if el is None:
                self.status_label.set_text("Failed to create a required GStreamer element")
                return
            self.pipeline.add(el)

        # Link common trunk
        src.link(parsebin)
        self.mux.link(tee)

        # Link Tee pads to branch queues
        tee_pad_up = tee.get_request_pad("src_%u")
        tee_pad_orig = tee.get_request_pad("src_%u")
        tee_pad_up.link(queue_up.get_static_pad("sink"))
        tee_pad_orig.link(queue_orig.get_static_pad("sink"))

        # Link upscaler branch
        upstream = queue_up
        for el in upscaler_elements + [postconv_up, sink_up]:
            if not upstream.link(el):
                print(f"Warning: could not link {upstream.get_name()} -> {el.get_name()}")
            upstream = el

        # Link original branch
        upstream = queue_orig
        #for el in [postconv_orig, capsfilter_orig, sink_orig]:
        for el in [postconv_orig, sink_orig]:
            if not upstream.link(el):
                print(f"Warning: could not link {upstream.get_name()} -> {el.get_name()}")
            upstream = el

        parsebin.connect("pad-added", self.on_parsebin_pad_added)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

        self.pipeline.set_state(Gst.State.PLAYING)

        mode = "Dual-window: Original + Upscaled" if self._upscaler_active() else "Dual-window: Decode-only"
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
            fakesink = Gst.ElementFactory.make("fakesink", None)
            fakesink.set_property("sync", False)
            self.pipeline.add(fakesink)
            fakesink.sync_state_with_parent()
            pad.link(fakesink.get_static_pad("sink"))

    def on_audio_decodebin_pad_added(self, decodebin, pad):
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

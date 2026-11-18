# src/utils/export.py
import os, glob, shutil, tempfile
import imageio.v2 as imageio
import polyscope as ps


def export_polyscope_animation(
    app,
    out_path="animation.mp4",
    fps=30,
    start_frame=0,  # Start frame
    end_frame=0,  # End frame
    transparent=False,  # Is the screenshot with a transparent background
    reset_first=False,  # Should the scene be reset first
):
    """
    Use Polyscope to take frame-by-frame screenshots and merge them into an MP4 file.
    Dependencies:imageio, polyscope(Python)
    Convention:DemoApp.callback() Advance one frame each time it is called (if applicable) app.playing=True)
    """
    # 1) Allow headless rendering (HPC) / SSH)
    # ps.set_allow_headless(True)

    # 2) Optional reset
    if reset_first:
        app.reset()

    # 3) Calculate the number of exported frames and the range
    total_frames = end_frame - start_frame + 1

    print(f"Exporting frames {start_frame} to {end_frame} (total: {total_frames})")
    print(f"Current frame: {app.fId}, Max frame: {app.max_fId}")

    # 4) Temporary frame directory
    tmp_dir = tempfile.mkdtemp(prefix="ps_frames_")
    frame_tmpl = os.path.join(tmp_dir, "frame_%06d.png")

    # 5) Should not be in playback state during export
    assert app.playing == False

    # 6) Frame-by-frame: Progress to specified frame → Screenshot
    try:
        # Export starting from the initial frame
        for i in range(total_frames):

            # Playback current frame
            app.fId = start_frame + i
            app.callback()

            # Screenshot (no pop-up, off-screen)
            ps.screenshot(filename=frame_tmpl % i, transparent_bg=transparent)

        # 7) Combine into MP4 ( H.264)
        print(f"Compositing {total_frames} frames to {out_path}")
        with imageio.get_writer(out_path, fps=fps, codec="libx264") as w:
            for png in sorted(glob.glob(os.path.join(tmp_dir, "frame_*.png"))):
                w.append_data(imageio.imread(png))

        print(f"Export completed successfully: {out_path}")

    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)

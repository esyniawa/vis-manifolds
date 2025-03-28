import os
import sys
import subprocess
import platform
from pathlib import Path
import tempfile
import shutil
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Union, Tuple
import logging

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


class EncoderType(Enum):
    """Enumeration of supported encoder types"""
    AUTO = auto()  # Automatically detect best available encoder
    NVIDIA = auto()  # NVIDIA NVENC hardware acceleration
    AMD = auto()  # AMD AMF hardware acceleration
    INTEL = auto()  # Intel QuickSync hardware acceleration
    VAAPI = auto()  # Video Acceleration API (Linux)
    SOFTWARE = auto()  # Software encoding (libx264)
    GIF = auto()  # Animated GIF (Pillow)
    FRAMES = auto()  # Export frames as images


class AnimationEncoder:
    """
    A versatile animation encoder class that handles different encoding methods
    with appropriate fallbacks based on system capabilities.
    """
    def __init__(self,
                 encoder_type: EncoderType = EncoderType.AUTO,
                 output_dir: Optional[str] = None,
                 quality: int = 23,  # Quality setting (lower = better, range: 0-51)
                 fps: int = 24,  # Frames per second
                 speed_preset: str = "medium",  # Encoding speed preset
                 bitrate: int = 2000,  # Bitrate in Kbps
                 threads: Optional[int] = None,  # CPU threads (None = auto)
                 verbose: bool = True):
        """
        Initialize the animation encoder with the specified settings.

        :param encoder_type: Type of encoder to use
        :param output_dir: Directory to save output files (default: current directory)
        :param quality: Video quality (CRF value, lower = better quality, range: 0-51)
        :param fps: Frames per second
        :param speed_preset: Encoding speed preset ('ultrafast' to 'veryslow')
        :param bitrate: Video bitrate in Kbps
        :param threads: Number of CPU threads to use (None = auto)
        :param verbose: Enable verbose logging
        """
        # Setup logging
        self.logger = logging.getLogger("AnimationEncoder")
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Store settings
        self.encoder_type = encoder_type
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.quality = quality
        self.fps = fps
        self.speed_preset = speed_preset
        self.bitrate = bitrate
        self.threads = threads
        self.verbose = verbose

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Detect system properties
        self.system = platform.system()  # 'Windows', 'Linux', 'Darwin'
        self.is_windows = self.system == 'Windows'
        self.is_linux = self.system == 'Linux'
        self.is_mac = self.system == 'Darwin'

        # Detect available encoders
        self._detect_encoders()

    def _detect_encoders(self) -> None:
        """Detect available encoders on the system"""
        self.available_encoders = {}

        # Check if FFmpeg is available
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if result.returncode == 0:
                self.ffmpeg_available = True
                self.logger.info(f"FFmpeg found: {result.stdout.splitlines()[0]}")

                # Check for hardware encoders
                self._detect_hardware_encoders()
            else:
                self.ffmpeg_available = False
                self.logger.warning("FFmpeg not available.")
        except FileNotFoundError:
            self.ffmpeg_available = False
            self.logger.warning("FFmpeg not found in system path.")

    def _detect_hardware_encoders(self) -> None:
        """Detect available hardware encoders"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-encoders"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            output = result.stdout

            # Define encoders to check for
            encoders_to_check = {
                "h264_nvenc": EncoderType.NVIDIA,
                "hevc_nvenc": EncoderType.NVIDIA,
                "h264_amf": EncoderType.AMD,
                "hevc_amf": EncoderType.AMD,
                "h264_qsv": EncoderType.INTEL,
                "hevc_qsv": EncoderType.INTEL,
                "h264_vaapi": EncoderType.VAAPI,
                "libx264": EncoderType.SOFTWARE,
            }

            for codec, encoder_type in encoders_to_check.items():
                if codec in output:
                    if encoder_type not in self.available_encoders:
                        self.available_encoders[encoder_type] = []
                    self.available_encoders[encoder_type].append(codec)
                    self.logger.info(f"Found encoder: {codec}")

            # Always add GIF as available
            self.available_encoders[EncoderType.GIF] = ["pillow"]
            self.available_encoders[EncoderType.FRAMES] = ["frames"]

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error detecting hardware encoders: {e}")

    def _get_best_encoder(self) -> Tuple[EncoderType, str]:
        """Determine the best available encoder based on preferences and availability"""
        # If specific encoder type requested, try to use that
        if self.encoder_type != EncoderType.AUTO and self.encoder_type in self.available_encoders:
            encoder_type = self.encoder_type
            codec = self.available_encoders[encoder_type][0]  # Use first available codec for this type
            return encoder_type, codec

        # If AUTO or requested encoder not available, find the best option
        # Preference order: NVIDIA > AMD > INTEL > VAAPI > SOFTWARE > GIF
        for encoder_type in [EncoderType.NVIDIA, EncoderType.AMD, EncoderType.INTEL,
                             EncoderType.VAAPI, EncoderType.SOFTWARE]:
            if encoder_type in self.available_encoders:
                codec = self.available_encoders[encoder_type][0]
                return encoder_type, codec

        # If no video encoders available, fall back to GIF
        return EncoderType.GIF, "pillow"

    def _get_ffmpeg_args(self, encoder_type: EncoderType, codec: str) -> Dict[str, Any]:
        """Get FFmpeg writer arguments based on encoder type and codec"""
        extra_args = []

        # Base arguments that apply to most encoders
        if encoder_type != EncoderType.GIF:
            extra_args.extend(['-pix_fmt', 'yuv420p'])  # Ensure compatibility

        # Hardware-specific options
        if encoder_type == EncoderType.NVIDIA:
            extra_args.extend([
                '-preset', 'p7' if 'hevc' in codec else 'fast',  # Use p7 preset for HEVC, fast for H264
                '-tune', 'hq',
                '-crf', str(self.quality)
            ])
        elif encoder_type == EncoderType.AMD:
            extra_args.extend([
                '-quality', 'speed',
                '-crf', str(self.quality)
            ])
        elif encoder_type == EncoderType.INTEL:
            extra_args.extend([
                '-preset', 'faster',
                '-crf', str(self.quality)
            ])
        elif encoder_type == EncoderType.VAAPI:
            extra_args.extend([
                '-vaapi_device', '/dev/dri/renderD128',
                '-vf', 'format=nv12,hwupload',
                '-qp', str(self.quality)
            ])
        elif encoder_type == EncoderType.SOFTWARE:
            extra_args.extend([
                '-preset', self.speed_preset,
                '-crf', str(self.quality)
            ])

            # Set thread count for software encoding
            if self.threads is not None:
                extra_args.extend(['-threads', str(self.threads)])

        return {
            'fps': self.fps,
            'bitrate': self.bitrate,
            'codec': codec,
            'extra_args': extra_args
        }

    def save_animation(self,
                       animation: FuncAnimation,
                       filename: str,
                       writer_kwargs: Optional[Dict[str, Any]] = None) -> str:
        """
        Save the animation using the best available encoder.
        """
        # Get best available encoder
        encoder_type, codec = self._get_best_encoder()
        self.logger.info(f"Using encoder: {encoder_type.name} ({codec})")

        # Ensure filename has no extension (we'll add the right one)
        filename = Path(filename).stem

        # Prepare output path
        output_path = self.output_dir / filename

        # Handle based on encoder type
        if encoder_type == EncoderType.GIF:
            return self._save_as_gif(animation, output_path, writer_kwargs)
        elif encoder_type == EncoderType.FRAMES:
            return self._save_as_frames(animation, output_path, writer_kwargs)
        else:
            return self._save_with_ffmpeg(animation, output_path, encoder_type, codec, writer_kwargs)

    def _save_with_ffmpeg(self,
                          animation: FuncAnimation,
                          output_path: Path,
                          encoder_type: EncoderType,
                          codec: str,
                          writer_kwargs: Optional[Dict[str, Any]] = None) -> str:
        """Save animation using FFmpeg with the specified encoder"""
        # Get FFmpeg arguments
        ffmpeg_args = self._get_ffmpeg_args(encoder_type, codec)

        # Override with any user-specified arguments
        if writer_kwargs:
            ffmpeg_args.update(writer_kwargs)

        # Extract extra_args from ffmpeg_args
        extra_args = ffmpeg_args.pop('extra_args', [])

        # Create the FFmpeg writer
        writer = FFMpegWriter(
            **ffmpeg_args,
            extra_args=extra_args
        )

        # Determine file extension based on codec
        if 'hevc' in codec or 'h264' in codec or 'nvenc' in codec or 'qsv' in codec or 'amf' in codec:
            extension = '.mp4'  # Most H.264/H.265 codecs use MP4 container
        elif 'vp9' in codec:
            extension = '.webm'  # VP9 typically uses WebM container
        elif 'av1' in codec:
            extension = '.mp4'  # AV1 can use MP4 container
        else:
            extension = '.mkv'  # Use MKV as a fallback for other codecs

        output_file = f"{output_path}{extension}"

        try:
            self.logger.info(f"Saving animation to {output_file} using {codec}...")
            animation.save(output_file, writer=writer)
            self.logger.info(f"Successfully saved animation to {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Error saving animation with {codec}: {e}")

            # Fall back to GIF if video encoding fails
            self.logger.info("Falling back to GIF format...")
            return self._save_as_gif(animation, output_path)

    def _save_as_gif(self,
                     animation: FuncAnimation,
                     output_path: Path,
                     writer_kwargs: Optional[Dict[str, Any]] = None) -> str:
        """Save animation as GIF using Pillow"""
        # Create the Pillow writer
        writer_args = {
            'fps': self.fps
        }

        # Override with any user-specified arguments
        if writer_kwargs:
            writer_args.update(writer_kwargs)

        writer = PillowWriter(**writer_args)

        output_file = f"{output_path}.gif"

        try:
            self.logger.info(f"Saving animation to {output_file} as GIF...")
            animation.save(output_file, writer=writer)
            self.logger.info(f"Successfully saved animation to {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Error saving animation as GIF: {e}")

            # Fall back to saving frames
            self.logger.info("Falling back to saving individual frames...")
            return self._save_as_frames(animation, output_path)

    def _save_as_frames(self,
                        animation: FuncAnimation,
                        output_path: Path,
                        writer_kwargs: Optional[Dict[str, Any]] = None) -> str:

        """Save animation as individual frames (last resort fallback)"""
        frames_dir = output_path.with_name(f"{output_path.name}_frames")
        frames_dir.mkdir(exist_ok=True)

        self.logger.info(f"Saving animation as individual frames to {frames_dir}...")

        # Get the figure from the animation
        fig = animation._fig

        # Get the frames
        frames = animation._save_seq

        try:
            # Save each frame
            for i, frame_data in enumerate(frames):
                # Get the artists for this frame
                frame_artists = animation._func(frame_data, *animation._args)

                if isinstance(frame_artists, tuple):
                    for artist in frame_artists:
                        if artist:
                            artist.set_animated(False)
                elif frame_artists:
                    frame_artists.set_animated(False)

                # Save the frame
                frame_path = frames_dir / f"frame_{i:04d}.png"
                fig.savefig(frame_path, dpi=100)

                if i % 10 == 0:
                    self.logger.info(f"Saved frame {i}/{len(frames)}")

            self.logger.info(f"Successfully saved {len(frames)} frames to {frames_dir}")
            return str(frames_dir)

        except Exception as e:
            self.logger.error(f"Error saving frames: {e}")
            return str(frames_dir)

    def list_available_encoders(self) -> None:
        """Print information about available encoders"""
        print("\nAvailable encoders:")
        print("---------------------")

        if not self.ffmpeg_available:
            print("- FFmpeg not available. Only GIF output supported.")
        else:
            if EncoderType.NVIDIA in self.available_encoders:
                print(f"NVIDIA NVENC: {', '.join(self.available_encoders[EncoderType.NVIDIA])}")
            else:
                print("NVIDIA NVENC: Not available")

            if EncoderType.AMD in self.available_encoders:
                print(f"AMD AMF: {', '.join(self.available_encoders[EncoderType.AMD])}")
            else:
                print("AMD AMF: Not available")

            if EncoderType.INTEL in self.available_encoders:
                print(f"Intel QuickSync: {', '.join(self.available_encoders[EncoderType.INTEL])}")
            else:
                print("Intel QuickSync: Not available")

            if EncoderType.VAAPI in self.available_encoders:
                print(f"VAAPI: {', '.join(self.available_encoders[EncoderType.VAAPI])}")
            else:
                print("VAAPI: Not available")

            if EncoderType.SOFTWARE in self.available_encoders:
                print(f"Software: {', '.join(self.available_encoders[EncoderType.SOFTWARE])}")
            else:
                print("Software encoding: Not available")

        print("GIF: Always available")
        print("Image Frames: Always available")

        # Print best encoder
        best_type, best_codec = self._get_best_encoder()
        print(f"\nBest available encoder: {best_type.name} ({best_codec})")
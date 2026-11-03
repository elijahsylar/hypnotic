"""Unit tests for the analysis and text paths. No audio device or display needed."""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hypnotic as h  # noqa: E402


def tone(freq, n=h.BLOCK_SIZE, amp=1.0):
    t = np.arange(n) / h.SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestProcessBlock:
    def test_bass_tone_lands_in_bass_band(self):
        a = h.AudioEngine()
        a.process_block(tone(200))
        assert a.bass > a.mids and a.bass > a.highs

    def test_high_tone_lands_in_high_band(self):
        a = h.AudioEngine()
        a.process_block(tone(15000))
        assert a.highs > a.bass

    def test_bands_are_clamped_and_fft_normalised(self):
        a = h.AudioEngine()
        a.process_block(tone(200, amp=100.0))
        assert 0.0 <= a.bass <= 1.0 and 0.0 <= a.mids <= 1.0 and 0.0 <= a.highs <= 1.0
        assert a.fft.max() <= 1.0 + 1e-9
        assert len(a.fft) == h.FFT_SIZE // 2

    def test_silence_is_quiet(self):
        a = h.AudioEngine()
        a.process_block(np.zeros(h.BLOCK_SIZE, dtype=np.float32))
        assert a.energy == pytest.approx(0.0, abs=1e-6)

    def test_peak_decays_after_a_hit(self):
        a = h.AudioEngine()
        a.process_block(tone(100))
        hit = a.peak
        a.process_block(np.zeros(h.BLOCK_SIZE, dtype=np.float32))
        assert 0 < a.peak < hit


class TestDemoSignal:
    def test_blocks_have_the_right_shape_and_are_deterministic(self):
        x, y = h.DemoSignal(seed=1), h.DemoSignal(seed=1)
        bx, by = x.next_block(), y.next_block()
        assert bx.shape == (h.BLOCK_SIZE,) and bx.dtype == np.float32
        assert np.array_equal(bx, by)

    def test_time_advances_between_blocks(self):
        d = h.DemoSignal()
        assert not np.array_equal(d.next_block(), d.next_block())


class TestKeywords:
    def test_basic_extraction_prefers_technical_terms(self):
        s = h.SpeechRecognizer()
        assert s.extract_keywords_basic("the fundamental derivation of probability") in (["derivation"], ["probability"])

    def test_basic_extraction_skips_short_words(self):
        s = h.SpeechRecognizer()
        assert s.extract_keywords_basic("a cat sat on it") == []

    def test_technical_word_rules(self):
        s = h.SpeechRecognizer()
        assert s.is_technical_word("integration")
        assert not s.is_technical_word("tion")


class TestHelpers:
    @pytest.mark.parametrize("hsv,rgb", [((0, 0, 0), (0, 0, 0)), ((0, 0, 1), (255, 255, 255)), ((0, 1, 1), (255, 0, 0))])
    def test_hsv_to_rgb(self, hsv, rgb):
        assert h.hsv_to_rgb(*hsv) == rgb

    def test_hsv_to_rgb_clamps(self):
        r, g, b = h.hsv_to_rgb(2.5, 9, -3)
        assert all(0 <= c <= 255 for c in (r, g, b))

    def test_parse_args_defaults(self):
        a = h.parse_args([])
        assert a.device is None and not a.demo and a.frames == 0

    def test_parse_args_device_and_flags(self):
        a = h.parse_args(["3", "--demo", "--frames", "10", "--effect", "2"])
        assert (a.device, a.demo, a.frames, a.effect) == (3, True, 10, 2)


def test_demo_run_renders_frames(tmp_path):
    h.main(["--demo", "--record", str(tmp_path), "--frames", "5"])
    assert len(list(tmp_path.glob("frame_*.png"))) == 5

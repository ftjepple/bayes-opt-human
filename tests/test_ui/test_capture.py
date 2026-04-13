"""Tests for the output capture utility."""

import sys

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import pytest

from bayesopt_human.ui.capture import CapturedOutput, capture_output


class TestCaptureOutput:
    """Tests for capture_output() context manager."""

    def test_captures_stdout(self):
        with capture_output() as captured:
            print("hello world")
        assert "hello world" in captured.text

    def test_captures_multiple_prints(self):
        with capture_output() as captured:
            print("line 1")
            print("line 2")
        assert "line 1" in captured.text
        assert "line 2" in captured.text

    def test_captures_figures(self):
        plt.close("all")
        with capture_output() as captured:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            plt.show()
        assert len(captured.figures) == 1
        assert captured.figures[0] is fig
        plt.close("all")

    def test_captures_multiple_figures(self):
        plt.close("all")
        with capture_output() as captured:
            fig1, ax1 = plt.subplots()
            ax1.plot([1, 2])
            plt.show()
            fig2, ax2 = plt.subplots()
            ax2.plot([3, 4])
            plt.show()
        assert len(captured.figures) == 2
        plt.close("all")

    def test_suppresses_show(self):
        """plt.show() should not raise or display anything."""
        with capture_output() as captured:
            fig, ax = plt.subplots()
            plt.show()  # should be a no-op
        assert len(captured.figures) == 1
        plt.close("all")

    def test_restores_stdout(self):
        original = sys.stdout
        with capture_output():
            pass
        assert sys.stdout is original

    def test_restores_stdout_on_exception(self):
        original = sys.stdout
        with pytest.raises(ValueError):
            with capture_output():
                raise ValueError("test error")
        assert sys.stdout is original

    def test_restores_plt_show(self):
        original_show = plt.show
        with capture_output():
            pass
        assert plt.show is original_show

    def test_restores_plt_show_on_exception(self):
        original_show = plt.show
        with pytest.raises(ValueError):
            with capture_output():
                raise ValueError("test error")
        assert plt.show is original_show

    def test_ignores_preexisting_figures(self):
        plt.close("all")
        pre_fig = plt.figure()
        with capture_output() as captured:
            new_fig = plt.figure()
        assert len(captured.figures) == 1
        assert captured.figures[0] is new_fig
        plt.close("all")

    def test_empty_context(self):
        with capture_output() as captured:
            pass
        assert captured.text == ""
        assert captured.figures == []

    def test_text_and_figures_combined(self):
        plt.close("all")
        with capture_output() as captured:
            print("some text")
            fig, ax = plt.subplots()
            ax.set_title("test")
            plt.show()
        assert "some text" in captured.text
        assert len(captured.figures) == 1
        plt.close("all")

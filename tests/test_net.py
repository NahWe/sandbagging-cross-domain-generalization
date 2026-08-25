import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from src.data.net import urlopen_with_retry


def _http_error(code):
    return urllib.error.HTTPError(url="http://x", code=code, msg="err", hdrs=None, fp=None)


def _response(body: bytes):
    response = MagicMock()
    response.read.return_value = body
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


@patch("src.data.net.time.sleep")
@patch("src.data.net.urllib.request.urlopen")
def test_succeeds_on_first_try_no_retry_needed(mock_urlopen, mock_sleep):
    mock_urlopen.return_value = _response(b"ok")
    assert urlopen_with_retry("http://x", timeout=5) == b"ok"
    mock_sleep.assert_not_called()


@patch("src.data.net.time.sleep")
@patch("src.data.net.urllib.request.urlopen")
def test_retries_on_5xx_then_succeeds(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = [_http_error(502), _http_error(503), _response(b"ok")]
    assert urlopen_with_retry("http://x", timeout=5) == b"ok"
    assert mock_sleep.call_count == 2


@patch("src.data.net.time.sleep")
@patch("src.data.net.urllib.request.urlopen")
def test_does_not_retry_on_4xx(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = _http_error(404)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urlopen_with_retry("http://x", timeout=5)
    assert exc_info.value.code == 404
    mock_sleep.assert_not_called()


@patch("src.data.net.time.sleep")
@patch("src.data.net.urllib.request.urlopen")
def test_gives_up_after_max_attempts(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = _http_error(502)
    with pytest.raises(urllib.error.HTTPError):
        urlopen_with_retry("http://x", timeout=5, max_attempts=3)
    assert mock_urlopen.call_count == 3
    assert mock_sleep.call_count == 2


@patch("src.data.net.time.sleep")
@patch("src.data.net.urllib.request.urlopen")
def test_retries_on_url_error(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = [urllib.error.URLError("timed out"), _response(b"ok")]
    assert urlopen_with_retry("http://x", timeout=5) == b"ok"
    assert mock_sleep.call_count == 1

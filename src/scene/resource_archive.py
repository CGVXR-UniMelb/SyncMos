"""Read selected files from the LINGO authors' public ZIP archives."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from shutil import copyfileobj
from zipfile import ZipFile

import requests
from fsspec.spec import AbstractBufferedFile


class DriveArchive(AbstractBufferedFile):
    """Expose HTTP byte ranges to ZipFile without downloading the whole archive."""

    def __init__(self, session, file_id):
        self.session = session
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        with self._request(url, 0, 1) as response:
            size = int(response.headers["Content-Range"].split("/")[-1])
        super().__init__(None, url, size=size, block_size=4 * 1024**2)

    def _request(self, url, start, end):
        response = self.session.get(
            url,
            headers={"Range": f"bytes={start}-{end-1}", "Accept-Encoding": "identity"},
            stream=True,
            timeout=(15, 120),
        )
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(
            f"bytes {start}-{end-1}/"
        ):
            response.close()
            raise RuntimeError(
                "Google Drive did not serve the requested byte range. Retry later or supply the official local ZIP in .env."
            )
        return response

    def _fetch_range(self, start, end):
        with self._request(self.path, start, end) as response:
            data = response.content
        if len(data) != end - start:
            raise OSError("Incomplete Google Drive range. Retry the preparation command.")
        return data


def copy_resource(stream, target):
    """Publish each file only after extraction and ZIP CRC verification succeed."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            copyfileobj(stream, output, length=4 * 1024**2)
            output.close()
            temporary.replace(target)
        finally:
            output.close()
            temporary.unlink(missing_ok=True)


def extract_resources(paths, file_id, local_zip=None):
    missing = [path for path in paths if not path.is_file() or not path.stat().st_size]
    if not missing:
        return
    with requests.Session() as session:
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=3))
        with (
            Path(local_zip).expanduser().open("rb") if local_zip else DriveArchive(session, file_id) as source,
            ZipFile(source) as archive,
        ):
            for path in missing:
                name = path.as_posix()
                matches = [member for member in archive.namelist() if member == name or member.endswith("/" + name)]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {name} in the official archive, found {len(matches)}.")
                print(f"Extracting {name}", flush=True)
                with archive.open(matches[0]) as stream:
                    copy_resource(stream, path)

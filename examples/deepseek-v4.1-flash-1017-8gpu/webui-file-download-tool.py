"""
title: File download
description: Give the user a download link for a file the code interpreter wrote.
version: 0.1.0
"""

# The code interpreter (Pyodide) runs in the user's browser, so files it writes
# to /mnt/uploads exist only in that page; Open WebUI v0.12.3 turns only PNG
# images in code output into stored links. This tool reads the file in the same
# browser workspace through the code interpreter's own event
# (``execute:python``), stores it in Open WebUI's file store, attaches it to
# the reply and returns a link that works on any device and after a reload.
# The bytes travel in base64 chunks below the 16 MB websocket buffer and never
# enter the model's context.

import base64
import io
import json
import mimetypes
import posixpath
import uuid

ROOT = "/mnt/uploads/"
CHUNK_BYTES = 4 * 1024 * 1024
_READ = r"""
import base64, json, os
path = {path!r}
size = os.path.getsize(path)
with open(path, "rb") as stream:
    stream.seek({offset})
    data = stream.read({length})
print("KAIRYU-FILE " + json.dumps({{"size": size, "data": base64.b64encode(data).decode()}}))
"""
mimetypes.add_type(
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"
)
mimetypes.add_type("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx")
mimetypes.add_type(
    "application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"
)


class Tools:
    async def provide_download(
        self,
        path: str,
        __request__=None,
        __user__=None,
        __event_call__=None,
        __event_emitter__=None,
        __metadata__=None,
    ) -> str:
        """
        Give the user a file you created or edited, as a download link. This is
        the only way the user can receive a file: after your code saves it under
        /mnt/uploads, call this with its path, then put the returned Markdown
        link in your reply exactly as written. Never write sandbox: links or
        file paths as links, and never describe screens, buttons or folders for
        downloading. In the code interpreter, python-docx (import docx),
        python-pptx (import pptx), openpyxl, xlsxwriter and pandas are
        available: import them with a plain import statement at the top of the
        code, which loads them (an importlib check reports them missing).
        :param path: Path of the file under /mnt/uploads, e.g. /mnt/uploads/plan.xlsx.
        :return: A Markdown download link to show the user.
        """
        path = posixpath.normpath(str(path or ""))
        if not path.startswith(ROOT) or path == ROOT.rstrip("/"):
            return f"Error: the file must be under {ROOT}; save it there first."
        if __event_call__ is None or __request__ is None or not __user__:
            return "Error: no browser session to read the file from; ask the user to retry."

        chunks: list[bytes] = []
        offset = 0
        size = None
        while size is None or offset < size:
            output = await __event_call__(
                {
                    "type": "execute:python",
                    "data": {
                        "id": str(uuid.uuid4()),
                        "code": _READ.format(path=path, offset=offset, length=CHUNK_BYTES),
                        "session_id": (__metadata__ or {}).get("session_id"),
                        "files": [],
                    },
                }
            )
            output = output if isinstance(output, dict) else {}
            stdout = output.get("stdout") or ""
            line = next(
                (row for row in stdout.splitlines() if row.startswith("KAIRYU-FILE ")), None
            )
            if line is None:
                detail = output.get("stderr") or output.get("error") or output
                return f"Error: could not read {path} in the code interpreter: {str(detail)[:300]}"
            payload = json.loads(line.removeprefix("KAIRYU-FILE "))
            size = int(payload["size"])
            data = base64.b64decode(payload["data"])
            if not data and offset < size:
                return f"Error: reading {path} stopped at byte {offset} of {size}."
            chunks.append(data)
            offset += len(data)
        content = b"".join(chunks)

        from fastapi import UploadFile
        from open_webui.models.users import Users
        from open_webui.routers.files import upload_file_handler

        name = posixpath.basename(path)
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        user = await Users.get_user_by_id(__user__["id"])
        stored = await upload_file_handler(
            __request__,
            file=UploadFile(
                file=io.BytesIO(content),
                filename=name,
                headers={"content-type": content_type},
            ),
            metadata={"source": "code_interpreter", "path": path},
            process=False,
            user=user,
        )
        url = f"/api/v1/files/{stored.id}/content?attachment=true"
        if __event_emitter__ is not None:
            await __event_emitter__(
                {
                    "type": "files",
                    "data": {
                        "files": [
                            {
                                "type": "file",
                                "id": stored.id,
                                "name": name,
                                "url": stored.id,
                                "size": len(content),
                                "content_type": content_type,
                            }
                        ]
                    },
                }
            )
        return (
            f"Stored {name} ({len(content)} bytes). Show the user this download link "
            f"exactly as written: [{name}]({url})"
        )

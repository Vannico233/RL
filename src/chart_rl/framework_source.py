"""Validate pip's installed-source record for the fixed upstream framework."""

from urllib.parse import urlsplit


def verify_framework_source(direct, config):
    url = direct.get("url", "")
    vcs = direct.get("vcs_info", {})
    archive = direct.get("archive_info", {})
    digest = archive.get("hashes", {}).get("sha256")
    if digest is None and archive.get("hash", "").startswith("sha256="):
        digest = archive["hash"].split("=", 1)[1]
    verified, kind = False, "unverified"
    if vcs:
        kind = "git"
        verified = (vcs.get("vcs") == "git" and vcs.get("commit_id") == config["framework_commit"]
                    and url.rstrip("/") in ("https://github.com/modelscope/ms-swift.git",
                                           "https://github.com/modelscope/ms-swift"))
    elif archive:
        kind = "source_archive"
        # A manually downloaded local archive is also valid, but only with
        # the exact same bytes. A version string alone is never sufficient.
        allowed_url = url.split("#", 1)[0] == config["framework_source_url"] or urlsplit(url).scheme == "file"
        verified = allowed_url and digest == config["framework_archive_sha256"]
    return {"verified": bool(verified), "kind": kind,
            "commit": config["framework_commit"] if verified else None,
            "archive_sha256": digest}

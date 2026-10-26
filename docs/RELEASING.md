# Releasing

This project publishes source and Python distributions through GitHub Releases.
It is not published to PyPI unless the project explicitly adopts that channel
in a future release.

## Release contract

- Release from a clean commit on the default branch and tag it `vX.Y.Z`.
- Treat `RELEASE_NOTES.md` as the source of truth for shipped changes. Move
  verified entries from `Unreleased` into a dated version section and leave a
  fresh `Unreleased` section behind.
- Update the version in `pyproject.toml`. Planned work, local patches, and
  unverified claims do not belong in a released section.
- Preserve the documented `trained_model.pt` format unless a replacement has a
  versioned, unit-tested loader compatibility path.
- Keep the library independent of Alpaca Desktop and `alpaca-pipelines`
  workflow, persistence, and UI contracts.

## Prepare and verify

1. Review the complete release diff and confirm that public documentation
   describes the version being shipped.
2. Run the required checks:

   ```bash
   python -m pytest
   python -m ruff check src tests
   python -m ruff format --check src tests
   ```

3. Build both distributions and inspect them before upload:

   ```bash
   python -m build
   python -m zipfile --list dist/bioacoustics_dl_toolbox-X.Y.Z-py3-none-any.whl
   tar -tzf dist/bioacoustics_dl_toolbox-X.Y.Z.tar.gz
   ```

4. Install the wheel into an isolated environment and verify the distribution
   version and import path. For local Alpaca Desktop development, reinstall the
   sibling checkout editable in that project's virtual environment and verify
   that imports resolve to this repository.

## Publish on GitHub

1. Commit the release preparation, push its branch, review it through a pull
   request, and merge it into the default branch.
2. Create and push an annotated `vX.Y.Z` tag at the verified default-branch
   release commit.
3. Create a GitHub Release from that tag using the corresponding
   `RELEASE_NOTES.md` section, and attach the wheel and source distribution.
   Do not upload either artifact to PyPI.
4. Confirm the GitHub Release is public, its assets download, the tag targets
   the intended commit, and the default branch is clean and synchronized.

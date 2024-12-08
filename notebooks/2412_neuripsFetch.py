"""Fetch openReview data and cache to `datadir`.

> uv run (name) (datadir)
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openreview-py",
# ]
# ///
from pathlib import Path
import pickle
import openreview
import json
import sys


def note2json(notes: list[openreview.Note]):
    for paper in notes:
        paper_data = {
            'title': paper.content['title']['value'],
            'tldr': paper.content.get('TLDR', {}).get('value'),
            'authors': paper.content['authors']['value'],
            'topic': paper.content['primary_area']['value'],
            'abstract': paper.content['abstract']['value'],
            'pdf_url': f"https://openreview.net/pdf?id={paper.id}"
        }
        yield paper_data


if __name__ == "__main__":
    openreview_id = 'NeurIPS.cc/2024/Conference/-/Submission'

    datadir = Path(sys.argv[1])
    _nameit = lambda x: datadir / f"2412_openreviewNotes{x}"
    pklf = _nameit('.pkl')
    if not pklf.exists():
        print(f"Storing to {pklf}")
        client = openreview.api.OpenReviewClient(
            'https://api2.openreview.net')

        submissions = client.get_all_notes(
            invitation=openreview_id,
            details='all'
        )
        with open(pklf, "wb") as f:
            pickle.dump(submissions, f)
    else:
        with open(pklf, "rb") as f:
            submissions = pickle.load(f)

    jsons = list(note2json(submissions))
    with open(_nameit('.json'), "w") as f:
        json.dump(jsons, f)
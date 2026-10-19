"""Resolve visible citations against observations, never against guessed filenames.

This verifies a visited position, not whether every attributed claim is true.
"""

import re

PAGE = re.compile(r"\[([^\[\]\n]+?):p(\d+)\]")
STAMP = r"(?:\d+:)?\d{1,2}:\d{2}(?:\.\d+)?"
MEDIA = re.compile(r"\[([^\[\]\n]+?):(" + STAMP + r")[–—-](" + STAMP + r")\]")


def seconds(stamp):
    parts = [float(p) for p in stamp.split(":")]
    if any(p >= 60 for p in parts[1:]):
        raise ValueError("invalid timestamp")
    return sum(part * 60**i for i, part in enumerate(reversed(parts)))


def resolve_references(answer, events, space_id=None):
    observations = [
        e["observation"]
        for e in events
        if e["type"] == "tool" and not e["observation"].get("error")
    ]
    # Search excerpts are observations too; they do not imply a whole page/image was read.
    for observation in list(observations):
        for hit in observation.get("hits", []):
            if hit.get("id") and hit.get("name"):
                excerpt = {**hit, "material_id": hit["id"], "evidence_kind": "search_excerpt"}
                if hit.get("page") and space_id:
                    excerpt["page_url"] = (
                        f"/api/spaces/{space_id}/materials/{hit['id']}/pages/{hit['page']}/image"
                    )
                observations.append(excerpt)
    references, seen = [], set()
    for kind, pattern in [("page", PAGE), ("media", MEDIA)]:
        for match in pattern.finditer(answer):
            label, name = match[0], match[1]
            if label in seen:
                continue
            seen.add(label)
            candidates = [o for o in observations if o.get("name") == name and o.get("material_id")]
            ids = {o["material_id"] for o in candidates}
            base = {"label": label, "name": name, "kind": kind, "position_verified": False}
            if len(ids) != 1:
                references.append({**base, "reason": "ambiguous_or_unobserved_material"})
                continue
            material_id = next(iter(ids))
            if kind == "page":
                page = int(match[2])
                observed = next(
                    (o for o in candidates if o.get("page") == page and o.get("page_url")), None
                )
                if observed:
                    references.append(
                        {
                            **base,
                            "material_id": material_id,
                            "page": page,
                            "position_verified": True,
                            "url": observed["page_url"],
                            "evidence_kind": observed.get("evidence_kind", "page_observation"),
                        }
                    )
                else:
                    references.append({**base, "reason": "page_not_observed"})
                continue
            try:
                start, end = seconds(match[2]), seconds(match[3])
            except ValueError:
                references.append({**base, "reason": "invalid_time"})
                continue
            ranges = sorted(
                (o["start_seconds"], o["end_seconds"])
                for o in candidates
                if "start_seconds" in o
                and "end_seconds" in o
                and end <= o.get("duration", o["end_seconds"])
            )
            cursor = start
            for a, b in ranges:
                if a <= cursor:
                    cursor = max(cursor, b)
            if start < end and cursor >= end:
                references.append(
                    {
                        **base,
                        "material_id": material_id,
                        "start_seconds": start,
                        "end_seconds": end,
                        "position_verified": True,
                        "url": f"/#media={material_id}&start={start}&end={end}",
                    }
                )
            else:
                references.append({**base, "reason": "time_range_not_observed"})
    return sorted(references, key=lambda ref: answer.index(ref["label"]))

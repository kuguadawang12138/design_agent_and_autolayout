# -*- coding: utf-8 -*-
"""圆弧加直线的多边形交面积。边界保持圆弧，不切成折线。"""
import math
from pathlib import Path

PI = math.pi
EPS = 1e-9
TOL = 1e-7


def _hypot(x: float, y: float) -> float:
    return math.hypot(x, y)


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def edge_point(edge: tuple, t: float) -> tuple[float, float]:
    if edge[0] == "L":
        _, x1, y1, x2, y2 = edge
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    _, cx, cy, radius, a0, a1 = edge
    ang = a0 + t * (a1 - a0)
    return (cx + radius * math.cos(ang), cy + radius * math.sin(ang))


def edge_area(edge: tuple) -> float:
    """闭合边界上这一段对 ∮ x dy 的贡献。"""
    if edge[0] == "L":
        _, x1, y1, x2, y2 = edge
        return (x1 + x2) * 0.5 * (y2 - y1)
    _, cx, cy, radius, a0, a1 = edge

    def antideriv(theta: float) -> float:
        return cx * radius * math.sin(theta) + radius * radius * (
            0.5 * theta + 0.5 * math.sin(theta) * math.cos(theta)
        )

    return antideriv(a1) - antideriv(a0)


def polygon_area(edges: list[tuple]) -> float:
    return sum(edge_area(edge) for edge in edges)


def _line_t(x1: float, y1: float, x2: float, y2: float, x: float, y: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    length2 = dx * dx + dy * dy
    if length2 <= EPS:
        return 0.0
    return ((x - x1) * dx + (y - y1) * dy) / length2


def _point_line_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    length = _hypot(dx, dy)
    if length <= EPS:
        return _hypot(px - x1, py - y1)
    return abs(_cross(px - x1, py - y1, dx, dy)) / length


def _angle_t(a0: float, a1: float, ang: float) -> float | None:
    """角度落在圆弧上时返回参数 t，否则返回空。"""
    sweep = a1 - a0
    if abs(sweep) < EPS:
        return None
    if sweep > 0:
        off = (ang - a0) % (2 * PI)
        if off > 2 * PI - 1e-8:
            off = 0.0
        if off <= sweep + 1e-8:
            return max(0.0, min(1.0, off / sweep))
        return None
    off = (a0 - ang) % (2 * PI)
    if off <= -sweep + 1e-8:
        return max(0.0, min(1.0, off / -sweep))
    return None


def _on_edge(px: float, py: float, edge: tuple) -> bool:
    if edge[0] == "L":
        _, x1, y1, x2, y2 = edge
        if _point_line_dist(px, py, x1, y1, x2, y2) > TOL:
            return False
        t = _line_t(x1, y1, x2, y2, px, py)
        return -1e-8 <= t <= 1 + 1e-8
    _, cx, cy, radius, a0, a1 = edge
    if abs(_hypot(px - cx, py - cy) - radius) > TOL:
        return False
    return _angle_t(a0, a1, math.atan2(py - cy, px - cx)) is not None


def _unique_ts(values: list[float]) -> list[float]:
    kept: list[float] = []
    for value in sorted(values):
        if EPS < value < 1 - EPS and all(abs(value - old) > 1e-8 for old in kept):
            kept.append(value)
    return kept


def _circle_line_hits(
    cx: float, cy: float, radius: float, x1: float, y1: float, x2: float, y2: float
) -> list[tuple[float, float, float]]:
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy
    aa = dx * dx + dy * dy
    if aa <= EPS:
        return []
    bb = 2 * (fx * dx + fy * dy)
    cc = fx * fx + fy * fy - radius * radius
    disc = bb * bb - 4 * aa * cc
    if disc < -1e-8:
        return []
    root = math.sqrt(max(0.0, disc))
    hits = []
    for sign in (-1.0, 1.0):
        t = (-bb + sign * root) / (2 * aa)
        if -1e-8 <= t <= 1 + 1e-8:
            hits.append((t, x1 + t * dx, y1 + t * dy))
        if root <= 1e-10:
            break
    return hits


def _circle_points(
    c1x: float, c1y: float, r1: float, c2x: float, c2y: float, r2: float
) -> list[tuple[float, float]]:
    dx, dy = c2x - c1x, c2y - c1y
    dist = _hypot(dx, dy)
    if dist <= EPS or dist > r1 + r2 + 1e-7 or dist < abs(r1 - r2) - 1e-7:
        return []
    along = (r1 * r1 - r2 * r2 + dist * dist) / (2 * dist)
    height2 = r1 * r1 - along * along
    if height2 < -1e-8:
        return []
    height = math.sqrt(max(0.0, height2))
    mx = c1x + along * dx / dist
    my = c1y + along * dy / dist
    ox = -dy * height / dist
    oy = dx * height / dist
    if height <= 1e-9:
        return [(mx, my)]
    return [(mx + ox, my + oy), (mx - ox, my - oy)]


def _params_on(edge: tuple, other: tuple) -> list[float]:
    if edge[0] == "L" and other[0] == "L":
        _, x1, y1, x2, y2 = edge
        _, x3, y3, x4, y4 = other
        den = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
        if abs(den) < 1e-12:
            if _point_line_dist(x3, y3, x1, y1, x2, y2) > TOL:
                return []
            if _point_line_dist(x4, y4, x1, y1, x2, y2) > TOL:
                return []
            return _unique_ts([
                _line_t(x1, y1, x2, y2, x3, y3),
                _line_t(x1, y1, x2, y2, x4, y4),
            ])
        t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / den
        u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / den
        if -1e-8 <= u <= 1 + 1e-8:
            return _unique_ts([t])
        return []
    if edge[0] == "L" and other[0] == "A":
        _, x1, y1, x2, y2 = edge
        _, cx, cy, radius, a0, a1 = other
        found = []
        for t, px, py in _circle_line_hits(cx, cy, radius, x1, y1, x2, y2):
            if _angle_t(a0, a1, math.atan2(py - cy, px - cx)) is not None:
                found.append(t)
        return _unique_ts(found)
    if edge[0] == "A" and other[0] == "L":
        _, cx, cy, radius, a0, a1 = edge
        _, x1, y1, x2, y2 = other
        found = []
        for _, px, py in _circle_line_hits(cx, cy, radius, x1, y1, x2, y2):
            t = _angle_t(a0, a1, math.atan2(py - cy, px - cx))
            if t is not None:
                found.append(t)
        return _unique_ts(found)
    _, c1x, c1y, r1, a0, a1 = edge
    _, c2x, c2y, r2, b0, b1 = other
    if _hypot(c1x - c2x, c1y - c2y) <= TOL and abs(r1 - r2) <= TOL:
        return _unique_ts([
            value
            for value in (_angle_t(a0, a1, b0), _angle_t(a0, a1, b1))
            if value is not None
        ])
    found = []
    for px, py in _circle_points(c1x, c1y, r1, c2x, c2y, r2):
        t = _angle_t(a0, a1, math.atan2(py - c1y, px - c1x))
        u = _angle_t(b0, b1, math.atan2(py - c2y, px - c2x))
        if t is not None and u is not None:
            found.append(t)
    return _unique_ts(found)


def _split(edge: tuple, params: list[float]) -> list[tuple]:
    cuts = [0.0, *params, 1.0]
    parts = []
    for left, right in zip(cuts, cuts[1:]):
        if right - left <= 1e-10:
            continue
        if edge[0] == "L":
            x1, y1 = edge_point(edge, left)
            x2, y2 = edge_point(edge, right)
            parts.append(("L", x1, y1, x2, y2))
        else:
            _, cx, cy, radius, a0, a1 = edge
            parts.append(("A", cx, cy, radius, a0 + left * (a1 - a0), a0 + right * (a1 - a0)))
    return parts


def _vertices(edges: list[tuple]) -> list[tuple[float, float]]:
    return [edge_point(edge, 0.0) for edge in edges]


def _ray_hits(px: float, py: float, edges: list[tuple]) -> int:
    for vx, vy in _vertices(edges):
        if abs(vy - py) < 1e-8:
            py += 1e-7
            break
    hits = 0
    for edge in edges:
        if edge[0] == "L":
            _, x1, y1, x2, y2 = edge
            if (y1 > py) == (y2 > py):
                continue
            t = (py - y1) / (y2 - y1)
            if x1 + t * (x2 - x1) > px:
                hits += 1
            continue
        _, cx, cy, radius, a0, a1 = edge
        disc = radius * radius - (py - cy) * (py - cy)
        if disc <= 1e-12:
            continue
        root = math.sqrt(disc)
        for sign in (-1.0, 1.0):
            ix = cx + sign * root
            if ix <= px + 1e-10:
                continue
            t = _angle_t(a0, a1, math.atan2(py - cy, ix - cx))
            if t is not None and EPS < t < 1 - EPS:
                hits += 1
    return hits


def _classify(px: float, py: float, edges: list[tuple]) -> str:
    for edge in edges:
        if _on_edge(px, py, edge):
            return "bound"
    if _ray_hits(px, py, edges) % 2 == 1:
        return "in"
    return "out"


def _midpoint(edge: tuple) -> tuple[float, float]:
    return edge_point(edge, 0.5)


def _key(point: tuple[float, float]) -> tuple[float, float]:
    return (round(point[0], 6), round(point[1], 6))


def _loop_area(fragments: list[tuple]) -> float | None:
    if not fragments:
        return 0.0
    starts: dict[tuple[float, float], list[int]] = {}
    for index, edge in enumerate(fragments):
        starts.setdefault(_key(edge_point(edge, 0.0)), []).append(index)
    used = [False] * len(fragments)
    total = 0.0
    for index in range(len(fragments)):
        if used[index]:
            continue
        area = 0.0
        cursor = index
        origin = _key(edge_point(fragments[cursor], 0.0))
        for _ in range(len(fragments) + 1):
            used[cursor] = True
            area += edge_area(fragments[cursor])
            end = _key(edge_point(fragments[cursor], 1.0))
            if end == origin:
                break
            nxt = next((item for item in starts.get(end, []) if not used[item]), None)
            if nxt is None:
                return None
            cursor = nxt
        else:
            return None
        total += area
    if not all(used):
        return None
    return total


def intersection_area(left: list[tuple], right: list[tuple]) -> float:
    """两个圆弧多边形的交集面积。组内边界方向要一致，外环逆时针。"""
    kept: list[tuple] = []
    for source, edges, other in (("A", left, right), ("B", right, left)):
        for edge in edges:
            params: list[float] = []
            for other_edge in other:
                params.extend(_params_on(edge, other_edge))
            for part in _split(edge, _unique_ts(params)):
                kind = _classify(*_midpoint(part), other)
                if kind == "in" or (kind == "bound" and source == "A"):
                    kept.append(part)
    area = _loop_area(kept)
    if area is None:
        return 0.0
    return area


def _bbox(edges: list[tuple]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for edge in edges:
        for t in (0.0, 1.0):
            x, y = edge_point(edge, t)
            xs.append(x)
            ys.append(y)
        if edge[0] == "A":
            _, cx, cy, radius, a0, a1 = edge
            sweep = a1 - a0
            for quarter in (0.0, 0.5 * PI, PI, 1.5 * PI, 2 * PI, -0.5 * PI, -PI, -1.5 * PI):
                t = _angle_t(a0, a1, quarter)
                if t is None:
                    continue
                xs.append(cx + radius * math.cos(quarter))
                ys.append(cy + radius * math.sin(quarter))
            if abs(sweep) > 0:
                xs.append(cx + radius * math.cos(a0))
                ys.append(cy + radius * math.sin(a0))
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_hit(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def multi_iou(left: list[list[tuple]], right: list[list[tuple]]) -> float:
    """两组内部互不重叠的异形。交集面积是两两交面积之和。"""
    area_a = sum(polygon_area(poly) for poly in left)
    area_b = sum(polygon_area(poly) for poly in right)
    boxes_b = [_bbox(poly) for poly in right]
    overlap = 0.0
    for poly in left:
        box = _bbox(poly)
        for other, other_box in zip(right, boxes_b):
            if _bbox_hit(box, other_box):
                overlap += intersection_area(poly, other)
    union = area_a + area_b - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def arc_from_three(
    sx: float, sy: float, mx: float, my: float, ex: float, ey: float
) -> tuple:
    """起点、弧上一点、终点还原圆心和有向圆弧。"""
    det = 2 * (sx * (my - ey) + mx * (ey - sy) + ex * (sy - my))
    if abs(det) < EPS:
        return ("L", sx, sy, ex, ey)
    cx = (
        (sx * sx + sy * sy) * (my - ey)
        + (mx * mx + my * my) * (ey - sy)
        + (ex * ex + ey * ey) * (sy - my)
    ) / det
    cy = (
        (sx * sx + sy * sy) * (ex - mx)
        + (mx * mx + my * my) * (sx - ex)
        + (ex * ex + ey * ey) * (mx - sx)
    ) / det
    radius = _hypot(sx - cx, sy - cy)
    a0 = math.atan2(sy - cy, sx - cx)
    am = math.atan2(my - cy, mx - cx)
    a1 = math.atan2(ey - cy, ex - cx)
    ccw_mid = (am - a0) % (2 * PI)
    ccw_end = (a1 - a0) % (2 * PI)
    if ccw_end < 1e-12:
        ccw_end = 2 * PI
    if ccw_mid <= ccw_end + 1e-8:
        return ("A", cx, cy, radius, a0, a0 + ccw_end)
    cw_end = (a0 - a1) % (2 * PI)
    if cw_end < 1e-12:
        cw_end = 2 * PI
    return ("A", cx, cy, radius, a0, a0 - cw_end)


def load_arc_polys(path: Path) -> list[list[tuple]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    pos = 0
    count = int(lines[pos])
    pos += 1
    polygons = []
    for _ in range(count):
        edges_n = int(lines[pos])
        pos += 1
        edges = []
        for _edge in range(edges_n):
            parts = lines[pos].split()
            pos += 1
            nums = [float(item) for item in parts[1:]]
            if parts[0] == "L":
                edges.append(("L", *nums))
            else:
                edges.append(arc_from_three(*nums))
        polygons.append(edges)
    return polygons


def _square(x0: float, y0: float, x1: float, y1: float) -> list[tuple]:
    return [
        ("L", x0, y0, x1, y0),
        ("L", x1, y0, x1, y1),
        ("L", x1, y1, x0, y1),
        ("L", x0, y1, x0, y0),
    ]


def reverse_edges(edges: list[tuple]) -> list[tuple]:
    """边序反过来，面积变号。"""
    out = []
    for edge in reversed(edges):
        if edge[0] == "L":
            _, x1, y1, x2, y2 = edge
            out.append(("L", x2, y2, x1, y1))
        else:
            _, cx, cy, radius, a0, a1 = edge
            out.append(("A", cx, cy, radius, a1, a0))
    return out


def ensure_ccw(edges: list[tuple]) -> list[tuple]:
    if edges and polygon_area(edges) < 0:
        return reverse_edges(edges)
    return edges


def loop_from_points(pts) -> list[tuple]:
    """折线环收成直线边。闭合重复点去掉。"""
    n = len(pts)
    if n >= 2:
        x0, y0 = float(pts[0][0]), float(pts[0][1])
        x1, y1 = float(pts[-1][0]), float(pts[-1][1])
        if abs(x0 - x1) <= 1e-8 and abs(y0 - y1) <= 1e-8:
            n -= 1
    edges: list[tuple] = []
    for i in range(n):
        x1, y1 = float(pts[i][0]), float(pts[i][1])
        j = (i + 1) % n
        x2, y2 = float(pts[j][0]), float(pts[j][1])
        if abs(x1 - x2) <= 1e-12 and abs(y1 - y2) <= 1e-12:
            continue
        edges.append(("L", x1, y1, x2, y2))
    if len(edges) < 3:
        return []
    return edges


def make_shape(outers: list[list[tuple]], holes: list[list[tuple]]) -> dict:
    """一份图形只建一次。外环、孔都收成逆时针，面积先算好。"""
    outers = [ensure_ccw(loop) for loop in outers if loop]
    holes = [ensure_ccw(loop) for loop in holes if loop]
    area = sum(polygon_area(loop) for loop in outers) - sum(polygon_area(loop) for loop in holes)
    if area < 0:
        area = 0.0
    return {
        "outers": outers,
        "holes": holes,
        "area": area,
        "obox": [_bbox(loop) for loop in outers],
        "hbox": [_bbox(loop) for loop in holes],
    }


def _loops_inter(left, left_boxes, right, right_boxes) -> float:
    if not left or not right:
        return 0.0
    total = 0.0
    for loop, box in zip(left, left_boxes):
        for other, other_box in zip(right, right_boxes):
            if _bbox_hit(box, other_box):
                total += intersection_area(loop, other)
    return total


def shapes_iou(query: dict, cand: dict) -> float:
    """一份对一份。孔用容斥从交集里扣掉。面积用建形状时算好的。"""
    area_a = query["area"]
    area_b = cand["area"]
    if area_a <= 0 and area_b <= 0:
        return 1.0
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter = (
        _loops_inter(query["outers"], query["obox"], cand["outers"], cand["obox"])
        - _loops_inter(query["outers"], query["obox"], cand["holes"], cand["hbox"])
        - _loops_inter(query["holes"], query["hbox"], cand["outers"], cand["obox"])
        + _loops_inter(query["holes"], query["hbox"], cand["holes"], cand["hbox"])
    )
    if inter < 0:
        inter = 0.0
    if inter > area_a:
        inter = area_a
    if inter > area_b:
        inter = area_b
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _rot_xy(x: float, y: float, k: int) -> tuple[float, float]:
    """绕原点顺时针 k*90°，和原来 shapely 的 -90*k 一致。"""
    if k == 1:
        return y, -x
    if k == 2:
        return -x, -y
    if k == 3:
        return -y, x
    return x, y


def rotate_loop(edges: list[tuple], k: int) -> list[tuple]:
    if k == 0:
        return edges
    ang = -k * PI / 2
    out = []
    for edge in edges:
        if edge[0] == "L":
            _, x1, y1, x2, y2 = edge
            ax, ay = _rot_xy(x1, y1, k)
            bx, by = _rot_xy(x2, y2, k)
            out.append(("L", ax, ay, bx, by))
        else:
            _, cx, cy, radius, a0, a1 = edge
            nx, ny = _rot_xy(cx, cy, k)
            out.append(("A", nx, ny, radius, a0 + ang, a1 + ang))
    return out


def rotate_shape(shape: dict, k: int) -> dict:
    """同一份边旋转，不重新生成图形。面积旋转不变。"""
    k %= 4
    if k == 0:
        return shape
    outers = [rotate_loop(loop, k) for loop in shape["outers"]]
    holes = [rotate_loop(loop, k) for loop in shape["holes"]]
    return {
        "outers": outers,
        "holes": holes,
        "area": shape["area"],
        "obox": [_bbox(loop) for loop in outers],
        "hbox": [_bbox(loop) for loop in holes],
    }


def _self_check() -> None:
    circle = [
        ("A", 0.0, 0.0, 10.0, 0.0, PI),
        ("A", 0.0, 0.0, 10.0, PI, 2 * PI),
    ]
    assert abs(polygon_area(circle) - PI * 100) < 1e-6
    assert _classify(0.0, 0.0, circle) == "in"
    assert _classify(20.0, 0.0, circle) == "out"
    assert abs(polygon_area(_square(0, 0, 2, 2)) - 4) < 1e-9
    assert abs(intersection_area(_square(0, 0, 2, 2), _square(1, 1, 3, 3)) - 1) < 1e-6
    assert abs(intersection_area(_square(0, 0, 2, 2), _square(1, 0, 3, 2)) - 2) < 1e-6
    assert abs(intersection_area(_square(0, 0, 1, 1), _square(1, 0, 2, 1))) < 1e-6
    assert abs(intersection_area(_square(0, 0, 1, 1), _square(2, 2, 3, 3))) < 1e-6
    inner = _square(0.25, 0.25, 0.75, 0.75)
    outer = _square(0, 0, 2, 2)
    assert abs(intersection_area(inner, outer) - polygon_area(inner)) < 1e-6
    stadium_a = Path(__file__).resolve().parent / "cases" / "stadium_20_a.txt"
    stadium_b = Path(__file__).resolve().parent / "cases" / "stadium_20_b.txt"
    if stadium_a.is_file() and stadium_b.is_file():
        one = load_arc_polys(stadium_a)[0]
        assert abs(polygon_area(one) - (864 + PI * 144)) < 1e-4
        other = load_arc_polys(stadium_b)[0]
        inter = intersection_area(one, other)
        assert abs(inter - (432 + PI * 144)) < 1e-3, inter
    same = make_shape([circle], [])
    assert abs(shapes_iou(same, same) - 1.0) < 1e-6
    moved = rotate_shape(same, 1)
    assert abs(shapes_iou(same, moved) - 1.0) < 1e-6
    hole = [
        ("A", 0.0, 0.0, 4.0, 0.0, PI),
        ("A", 0.0, 0.0, 4.0, PI, 2 * PI),
    ]
    donut = make_shape([circle], [hole])
    assert abs(donut["area"] - PI * (100 - 16)) < 1e-4
    assert abs(shapes_iou(donut, same) - donut["area"] / same["area"]) < 1e-4


if __name__ == "__main__":
    _self_check()
    print("arc_iou ok")

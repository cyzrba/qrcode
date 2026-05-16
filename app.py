import os
import tempfile
import time
from pathlib import Path

import av
import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from detector import decode_qr, decode_qr_adaptive, decode_qr_frame, draw_results

CATEGORIES = [
    "nominal", "blurred", "bright_spots", "brightness", "close",
    "curved", "damaged", "glare", "high_version", "lots",
    "monitor", "noncompliant", "pathological", "perspective",
    "rotations", "shadows",
]

CATEGORY_CN = {
    "nominal": "正常", "blurred": "模糊", "bright_spots": "亮斑",
    "brightness": "亮度异常", "close": "特写", "curved": "弯曲",
    "damaged": "损坏", "glare": "眩光", "high_version": "高版本",
    "lots": "多码", "monitor": "屏幕", "noncompliant": "非标准",
    "pathological": "极端", "perspective": "透视", "rotations": "旋转",
    "shadows": "阴影",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
.stApp {
    background: linear-gradient(135deg, #0e1117 0%, #161b22 50%, #0d1117 100%);
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(0, 240, 255, 0.12);
}

/* ── Title gradient ── */
.gradient-title {
    background: linear-gradient(135deg, #00f0ff 0%, #7c3aed 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.gradient-sub {
    color: rgba(255,255,255,0.45);
    font-size: 0.82rem;
    font-weight: 300;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* ── KPI metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,240,255,0.15);
    border-radius: 12px;
    padding: 16px 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0,240,255,0.06);
}
[data-testid="stMetric"] label {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #00f0ff !important;
    font-weight: 600;
}

/* ── Result card ── */
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,240,255,0.12);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    backdrop-filter: blur(8px);
}
.result-card h4 {
    color: #e2e8f0;
    margin: 0 0 8px 0;
    font-weight: 500;
}
.badge-ok {
    background: rgba(16,185,129,0.15);
    color: #34d399;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-fail {
    background: rgba(239,68,68,0.15);
    color: #f87171;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.data-text {
    color: #a5b4fc;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    background: rgba(99,102,241,0.08);
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid rgba(99,102,241,0.15);
    word-break: break-all;
}

/* ── Section header ── */
.section-header {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 600;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(0,240,255,0.12);
    margin: 1.5rem 0 1rem 0;
}

/* ── Tables ── */
.dataframe thead tr th {
    background: linear-gradient(135deg, rgba(0,240,255,0.08), rgba(124,58,237,0.08)) !important;
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}
.dataframe tbody tr:nth-child(even) {
    background: rgba(255,255,255,0.02) !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(0,240,255,0.1) !important;
}

/* ── Expander ── */
details {
    border: 1px solid rgba(0,240,255,0.1) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.02) !important;
}
details summary {
    color: #e2e8f0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00f0ff 0%, #7c3aed 100%);
    color: #0e1117;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 2rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    box-shadow: 0 0 25px rgba(0,240,255,0.3);
    transform: translateY(-1px);
}
</style>
"""


def _detect_single(img_bytes: bytes, filename: str, category: str | None) -> dict:
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name
    try:
        t0 = time.perf_counter()
        if category:
            results = decode_qr_adaptive(tmp_path, category)
        else:
            results = decode_qr(tmp_path)
        elapsed = time.perf_counter() - t0
        image = cv2.imread(tmp_path)
        annotated = draw_results(image, results) if results is not None and len(results) > 0 else image
        return {
            "filename": filename,
            "results": results or [],
            "annotated": annotated,
            "original": image,
            "elapsed": elapsed,
            "ok": len(results) > 0 if results else False,
        }
    finally:
        os.unlink(tmp_path)


def _render_kpi(total: int, detected: int, qr_count: int, avg_time: float):
    c1, c2, c3, c4 = st.columns(4)
    rate = detected / total * 100 if total else 0
    c1.metric("图片总数", total)
    c2.metric("检测成功", f"{detected}/{total}")
    c3.metric("检测率", f"{rate:.1f}%")
    c4.metric("QR 码总数", qr_count)


def _render_single_result(r: dict):
    col_img, col_info = st.columns([3, 2])
    with col_img:
        st.image(cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_info:
        st.markdown(f"### `{r['filename']}`")
        if r["ok"]:
            st.markdown(f'<span class="badge-ok">检测成功</span> &nbsp; {len(r["results"])} 个 QR 码 &nbsp;|&nbsp; {r["elapsed"]*1000:.0f}ms', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-fail">未检测到</span> &nbsp; {r["elapsed"]*1000:.0f}ms', unsafe_allow_html=True)
        for i, qr in enumerate(r["results"], 1):
            st.markdown(f'<div class="data-text">[{i}] {qr["data"]}</div>', unsafe_allow_html=True)
            st.caption(f"类型: {qr['type']}  |  位置: ({qr['rect']['x']}, {qr['rect']['y']}, {qr['rect']['w']}×{qr['rect']['h']})")


def _render_batch_results(all_results: list[dict]):
    total = len(all_results)
    detected = sum(1 for r in all_results if r["ok"])
    qr_count = sum(len(r["results"]) for r in all_results)
    avg_time = sum(r["elapsed"] for r in all_results) / total if total else 0

    st.markdown('<div class="section-header">数据看板</div>', unsafe_allow_html=True)
    _render_kpi(total, detected, qr_count, avg_time)
    st.markdown("")

    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig_pie = go.Figure(data=[go.Pie(
            labels=["检测成功", "未检测到"],
            values=[detected, total - detected],
            hole=0.55,
            marker=dict(colors=["#10b981", "#ef4444"]),
            textinfo="label+percent",
            textfont=dict(size=13, color="#e2e8f0"),
        )])
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=30, b=30, l=30, r=30),
            height=320,
            title=dict(text="检测结果分布", font=dict(size=14, color="#94a3b8")),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        file_labels = [r["filename"][:18] for r in all_results]
        file_status = [1 if r["ok"] else 0 for r in all_results]
        colors = ["#10b981" if s else "#ef4444" for s in file_status]
        fig_bar = go.Figure(data=[go.Bar(
            x=file_labels,
            y=[len(r["results"]) for r in all_results],
            marker=dict(color=colors, line=dict(width=0)),
            hovertext=[f'{r["filename"]}<br>{"成功" if r["ok"] else "失败"}<br>{len(r["results"])} 个QR码' for r in all_results],
            hoverinfo="text",
        )])
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="QR 码数量", gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(t=30, b=60, l=40, r=20),
            height=320,
            title=dict(text="每张图片 QR 码数量", font=dict(size=14, color="#94a3b8")),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-header">逐图检测结果</div>', unsafe_allow_html=True)
    for r in all_results:
        label = f"{'✅' if r['ok'] else '❌'} &nbsp; {r['filename']} &nbsp;—&nbsp; {len(r['results'])} 个QR码 &nbsp;|&nbsp; {r['elapsed']*1000:.0f}ms"
        with st.expander(label, expanded=False):
            c1, c2 = st.columns([2, 3])
            with c1:
                st.image(cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                if r["results"]:
                    for i, qr in enumerate(r["results"], 1):
                        st.markdown(f'<div class="data-text">[{i}] {qr["data"]}</div>', unsafe_allow_html=True)
                        st.caption(f"类型: {qr['type']}  |  位置: ({qr['rect']['x']}, {qr['rect']['y']}, {qr['rect']['w']}×{qr['rect']['h']})")
                else:
                    st.info("该图片未检测到 QR 码")

    rows = []
    for r in all_results:
        data_str = " | ".join(q["data"][:80] for q in r["results"]) if r["results"] else "—"
        rows.append({
            "文件名": r["filename"],
            "状态": "✅" if r["ok"] else "❌",
            "QR数量": len(r["results"]),
            "解码内容": data_str,
            "耗时(ms)": f"{r['elapsed']*1000:.0f}",
        })
    st.markdown('<div class="section-header">详情表格</div>', unsafe_allow_html=True)
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_batch_by_category(all_results: list[dict]):
    cat_map: dict[str, list[dict]] = {}
    for r in all_results:
        cat = r.get("category", "unknown")
        cat_map.setdefault(cat, []).append(r)

    total = len(all_results)
    detected = sum(1 for r in all_results if r["ok"])
    qr_count = sum(len(r["results"]) for r in all_results)
    avg_time = sum(r["elapsed"] for r in all_results) / total if total else 0

    st.markdown('<div class="section-header">总览看板</div>', unsafe_allow_html=True)
    _render_kpi(total, detected, qr_count, avg_time)
    st.markdown("")

    col_pie, col_cat = st.columns(2)

    with col_pie:
        fig = go.Figure(data=[go.Pie(
            labels=["检测成功", "未检测到"],
            values=[detected, total - detected],
            hole=0.55,
            marker=dict(colors=["#10b981", "#ef4444"]),
            textinfo="label+percent",
            textfont=dict(size=13, color="#e2e8f0"),
        )])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=30, b=30, l=30, r=30), height=320,
            title=dict(text="整体检测分布", font=dict(size=14, color="#94a3b8")),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_cat:
        cat_names = sorted(cat_map.keys())
        cat_detected = [sum(1 for r in cat_map[c] if r["ok"]) for c in cat_names]
        cat_total = [len(cat_map[c]) for c in cat_names]
        cat_rate = [d / t * 100 if t else 0 for d, t in zip(cat_detected, cat_total)]
        cat_labels = [f"{CATEGORY_CN.get(c, c)}<br>{c}" for c in cat_names]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=cat_labels, y=cat_rate,
            marker=dict(
                color=cat_rate,
                colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
                cmin=0, cmax=100,
                line=dict(width=0),
            ),
            text=[f"{r:.0f}%" for r in cat_rate],
            textposition="outside", textfont=dict(size=10, color="#e2e8f0"),
            hovertext=[f"{c}<br>{d}/{t} ({r:.1f}%)" for c, d, t, r in zip(cat_names, cat_detected, cat_total, cat_rate)],
            hoverinfo="text",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="检测率 %", gridcolor="rgba(255,255,255,0.04)", range=[0, 110]),
            margin=dict(t=30, b=80, l=40, r=20), height=320,
            title=dict(text="各类别检测率", font=dict(size=14, color="#94a3b8")),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    for cat in sorted(cat_map.keys()):
        items = cat_map[cat]
        cat_det = sum(1 for r in items if r["ok"])
        cat_tot = len(items)
        cat_pct = cat_det / cat_tot * 100 if cat_tot else 0
        cn = CATEGORY_CN.get(cat, cat)
        header = f"{'🟢' if cat_pct >= 80 else '🟡' if cat_pct >= 50 else '🔴'} {cn} ({cat}) — {cat_det}/{cat_tot} ({cat_pct:.1f}%)"
        with st.expander(header, expanded=False):
            for r in items:
                badge = "✅" if r["ok"] else "❌"
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.image(cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB), use_container_width=True)
                with c2:
                    st.markdown(f"**{r['filename']}** &nbsp; {badge} &nbsp; {len(r['results'])} 个QR码 &nbsp;|&nbsp; {r['elapsed']*1000:.0f}ms")
                    if r["results"]:
                        for i, qr in enumerate(r["results"], 1):
                            st.markdown(f'<div class="data-text">[{i}] {qr["data"]}</div>', unsafe_allow_html=True)
                    else:
                        st.info("该图片未检测到 QR 码")

    rows = []
    for r in all_results:
        data_str = " | ".join(q["data"][:80] for q in r["results"]) if r["results"] else "—"
        rows.append({
            "类别": r.get("category", "—"),
            "文件名": r["filename"],
            "状态": "✅" if r["ok"] else "❌",
            "QR数量": len(r["results"]),
            "解码内容": data_str,
            "耗时(ms)": f"{r['elapsed']*1000:.0f}",
        })
    st.markdown('<div class="section-header">详情表格</div>', unsafe_allow_html=True)
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_camera_mode():
    st.markdown('<div class="section-header">摄像头实时检测</div>', unsafe_allow_html=True)

    if "cam_history" not in st.session_state:
        st.session_state.cam_history = []
    if "cam_frame_count" not in st.session_state:
        st.session_state.cam_frame_count = 0
    if "cam_detect_count" not in st.session_state:
        st.session_state.cam_detect_count = 0

    class QRProcessor(VideoProcessorBase):
        def __init__(self):
            self.last_results = []
            self.last_annotated = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            results = decode_qr_frame(img)
            annotated = draw_results(img, results) if results else img
            self.last_results = results
            self.last_annotated = annotated
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    ctx = webrtc_streamer(
        key="qr-camera",
        video_processor_factory=QRProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown("")
    result_placeholder = st.empty()
    stats_placeholder = st.empty()
    history_placeholder = st.empty()

    while ctx.state.playing:
        if ctx.video_processor:
            proc = ctx.video_processor
            results = proc.last_results
            st.session_state.cam_frame_count += 1
            if results:
                st.session_state.cam_detect_count += 1
                entry = {
                    "time": time.strftime("%H:%M:%S"),
                    "data": results[0]["data"][:80],
                    "count": len(results),
                }
                st.session_state.cam_history.insert(0, entry)
                st.session_state.cam_history = st.session_state.cam_history[:20]

            with stats_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("总帧数", st.session_state.cam_frame_count)
                c2.metric("检测到的帧", st.session_state.cam_detect_count)
                rate = st.session_state.cam_detect_count / st.session_state.cam_frame_count * 100 if st.session_state.cam_frame_count else 0
                c3.metric("实时检测率", f"{rate:.1f}%")

            with result_placeholder.container():
                if results:
                    for i, qr in enumerate(results, 1):
                        st.markdown(f'<div class="data-text">[{i}] {qr["data"]}</div>', unsafe_allow_html=True)
                else:
                    st.caption("等待 QR 码...")

            with history_placeholder.container():
                if st.session_state.cam_history:
                    st.markdown("**最近解码记录**")
                    for h in st.session_state.cam_history[:8]:
                        st.caption(f"`{h['time']}` — {h['count']} 个QR码: {h['data']}")

        time.sleep(0.3)

    if not ctx.state.playing and st.session_state.cam_frame_count > 0:
        with stats_placeholder.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("总帧数", st.session_state.cam_frame_count)
            c2.metric("检测到的帧", st.session_state.cam_detect_count)
            rate = st.session_state.cam_detect_count / st.session_state.cam_frame_count * 100 if st.session_state.cam_frame_count else 0
            c3.metric("最终检测率", f"{rate:.1f}%")
        st.info("摄像头已停止。点击上方按钮重新启动。")


def _render_video_mode(uploaded_file):
    st.markdown('<div class="section-header">视频检测</div>', unsafe_allow_html=True)

    cache_key = f"video_{uploaded_file.name}_{uploaded_file.size}"
    needs_process = (
        cache_key not in st.session_state
        or st.session_state.get(f"{cache_key}_name") != uploaded_file.name
    )

    if needs_process:
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            container = av.open(tmp_path)
            stream = container.streams.video[0]
            video_fps = float(stream.average_rate) if stream.average_rate else 30.0
            total_frames = stream.frames if stream.frames > 0 else int(stream.duration * video_fps / av.time_base if stream.duration else 0)
            duration = float(stream.duration / av.time_base) if stream.duration else 0
            width, height = stream.width, stream.height

            process_step = max(1, int(video_fps / 5))

            st.caption(f"文件: `{uploaded_file.name}` | {duration:.1f}s | {video_fps:.0f}fps | {width}×{height}")
            progress = st.progress(0, text="正在逐帧检测...")

            frame_results = []
            frame_idx = 0
            decoded_count = 0

            for frame in container.decode(video=0):
                if frame_idx % process_step == 0:
                    timestamp = frame_idx / video_fps
                    img = frame.to_ndarray(format="bgr24")
                    results = decode_qr_frame(img)
                    ok = len(results) > 0
                    if ok:
                        decoded_count += 1
                    frame_results.append({
                        "frame": frame_idx,
                        "time": timestamp,
                        "ok": ok,
                        "count": len(results),
                        "data": " | ".join(q["data"][:60] for q in results) if results else "",
                        "results": results,
                    })
                    processed = len(frame_results)
                    est_total = max(processed, total_frames // process_step) if total_frames > 0 else processed + 1
                    progress.progress(
                        min(processed / est_total, 1.0),
                        text=f"帧 {frame_idx} ({timestamp:.1f}s) — 已解码 {decoded_count}/{processed}",
                    )
                frame_idx += 1

            progress.empty()
            container.close()

            st.session_state[cache_key] = {
                "frame_results": frame_results,
                "video_fps": video_fps,
                "duration": duration,
                "width": width,
                "height": height,
                "tmp_path": tmp_path,
                "decoded_count": decoded_count,
            }
            st.session_state[f"{cache_key}_name"] = uploaded_file.name

        except Exception as e:
            st.error(f"视频处理出错: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return

    cached = st.session_state[cache_key]
    frame_results = cached["frame_results"]
    video_fps = cached["video_fps"]
    duration = cached["duration"]
    width = cached["width"]
    height = cached["height"]
    tmp_path = cached["tmp_path"]

    st.markdown("")
    fps_sample = st.slider("抽样显示帧率 (帧/秒)", 1, 10, 2, key="video_fps_slider")
    show_step = max(1, int(video_fps / fps_sample))
    show_results = [r for i, r in enumerate(frame_results) if i % max(1, show_step // max(1, int(video_fps / 5))) == 0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("视频时长", f"{duration:.1f}s")
    c2.metric("原始帧率", f"{video_fps:.0f} fps")
    c3.metric("分辨率", f"{width}×{height}")
    c4.metric("抽样帧率", f"{fps_sample} 帧/秒")

    st.markdown('<div class="section-header">时间线</div>', unsafe_allow_html=True)
    times = [r["time"] for r in show_results]
    counts = [r["count"] for r in show_results]
    colors_tl = ["#10b981" if r["ok"] else "#ef4444" for r in show_results]
    fig_tl = go.Figure(data=[go.Bar(
        x=times, y=counts,
        marker=dict(color=colors_tl, line=dict(width=0)),
        hovertext=[f'{r["time"]:.1f}s<br>{"✅ " + str(r["count"]) + " 个QR码" if r["ok"] else "❌ 未检测到"}' for r in show_results],
        hoverinfo="text",
    )])
    fig_tl.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(title="时间 (秒)", gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="QR 码数量", gridcolor="rgba(255,255,255,0.04)"),
        margin=dict(t=20, b=40, l=40, r=20), height=220,
        showlegend=False,
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown('<div class="section-header">检测总览</div>', unsafe_allow_html=True)
    total_sampled = len(frame_results)
    decoded_count = cached["decoded_count"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("抽样帧数", total_sampled)
    k2.metric("检测到 QR 的帧", decoded_count)
    k3.metric("帧检测率", f"{decoded_count/total_sampled*100:.1f}%" if total_sampled else "0%")
    unique_data = set()
    for r in frame_results:
        for q in r["results"]:
            unique_data.add(q["data"])
    k4.metric("去重 QR 码数", len(unique_data))

    detected_frames = [r for r in frame_results if r["ok"]]
    if detected_frames:
        st.markdown('<div class="section-header">关键帧（检测到 QR 码）</div>', unsafe_allow_html=True)

        display_frames = detected_frames
        if len(display_frames) > 20:
            indices = np.linspace(0, len(display_frames) - 1, 20, dtype=int)
            display_frames = [display_frames[i] for i in indices]
            st.caption(f"共 {len(detected_frames)} 个关键帧，展示 20 个采样帧")

        for r in display_frames:
            label = f"⏱ {r['time']:.1f}s — 帧 {r['frame']} — {r['count']} 个QR码: {r['data'][:50]}"
            with st.expander(label, expanded=False):
                if os.path.exists(tmp_path):
                    cap = av.open(tmp_path)
                    cap.seek(int(r["time"] * av.time_base), stream=cap.streams.video[0])
                    for fr in cap.decode(video=0):
                        img_bgr = fr.to_ndarray(format="bgr24")
                        annotated = draw_results(img_bgr, r["results"])
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                        break
                    cap.close()
                for i, qr in enumerate(r["results"], 1):
                    st.markdown(f'<div class="data-text">[{i}] {qr["data"]}</div>', unsafe_allow_html=True)
                    st.caption(f"类型: {qr['type']}  |  位置: ({qr['rect']['x']}, {qr['rect']['y']}, {qr['rect']['w']}×{qr['rect']['h']})")

    st.markdown('<div class="section-header">解码记录表</div>', unsafe_allow_html=True)
    rows = []
    for r in frame_results:
        if r["ok"]:
            rows.append({
                "时间": f"{r['time']:.2f}s",
                "帧号": r["frame"],
                "QR数量": r["count"],
                "解码内容": r["data"][:100],
            })
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.warning("整个视频中未检测到任何 QR 码。")


def main():
    st.set_page_config(
        page_title="QR 码智能检测系统",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="gradient-title">QR 码检测</div>', unsafe_allow_html=True)
        st.markdown('<div class="gradient-sub">自适应预处理引擎</div>', unsafe_allow_html=True)

        mode = st.radio("上传模式", ["单张图片", "批量上传", "视频检测", "摄像头实时"], horizontal=True)
        st.markdown("")

        category = st.selectbox(
            "检测类别",
            options=["(auto)"] + CATEGORIES,
            format_func=lambda x: f"{CATEGORY_CN.get(x, x)} ({x})" if x != "(auto)" else "自动识别 — 基础灰度解码",
        )
        sel_cat = None if category == "(auto)" else category

        st.markdown("---")

        if mode == "单张图片":
            uploaded = st.file_uploader(
                "上传图片",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
                accept_multiple_files=False,
            )
        elif mode == "视频检测":
            uploaded = st.file_uploader(
                "上传视频",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                accept_multiple_files=False,
            )
        else:
            uploaded = st.file_uploader(
                "上传多张图片",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
                accept_multiple_files=True,
            )

        run_btn = st.button("开始检测", use_container_width=True, type="primary")

    if mode == "摄像头实时":
        _render_camera_mode()
    elif mode == "视频检测":
        video_cache_key = f"video_{uploaded.name}_{uploaded.size}" if uploaded else None
        has_cached = video_cache_key and video_cache_key in st.session_state

        if uploaded is not None and (run_btn or has_cached):
            _render_video_mode(uploaded)
        elif uploaded is None:
            st.markdown("")
            st.info("请在侧边栏上传视频文件，然后点击 **开始检测**。")
            with st.expander("视频检测说明", expanded=True):
                st.markdown("""
                1. **上传** 包含 QR 码的视频文件（MP4/AVI/MOV/MKV/WebM）
                2. 系统自动按间隔**抽帧检测**（默认每秒 2 帧）
                3. 查看 **时间线图表**：每帧检测结果随时间变化
                4. 查看 **关键帧**：标注了 QR 码的帧画面
                5. 查看 **解码记录表**：所有检测到的 QR 码及对应时间戳
                """)
    elif mode == "单张图片":
        if uploaded is not None and run_btn:
            with st.spinner("正在解码 QR 码..."):
                result = _detect_single(uploaded.getvalue(), uploaded.name, sel_cat)
            st.markdown('<div class="section-header">检测结果</div>', unsafe_allow_html=True)
            _render_single_result(result)
        elif uploaded is None:
            st.markdown("")
            st.info("请在侧边栏上传图片，然后点击 **开始检测**。")
            with st.expander("使用说明", expanded=True):
                st.markdown("""
                1. **上传** 包含 QR 码的图片
                2. **选择** 检测类别以启用自适应预处理（或保持「自动识别」）
                3. **点击** 开始检测
                4. 查看标注图片和解码数据

                **类别说明：**
                - `自动识别` — 基础灰度解码，速度最快
                - `正常` — 光线良好、正面拍摄
                - `模糊` — 运动模糊或散焦模糊
                - `眩光` — 反光表面
                - `特写` — 近距离 / 大尺寸 QR 码
                - `弯曲` — 曲面或弯折的 QR 码
                """)
    else:
        if uploaded and run_btn:
            all_results = []
            progress = st.progress(0, text="正在处理图片...")
            for i, f in enumerate(uploaded):
                progress.progress((i + 1) / len(uploaded), text=f"处理中: {f.name} ({i+1}/{len(uploaded)})")
                r = _detect_single(f.getvalue(), f.name, sel_cat)
                all_results.append(r)
            progress.empty()
            st.markdown('<div class="section-header">批量检测结果</div>', unsafe_allow_html=True)
            _render_batch_results(all_results)
        elif not uploaded:
            st.markdown("")
            st.info("请在侧边栏上传多张图片，然后点击 **开始检测**。")
            with st.expander("批量模式说明", expanded=True):
                st.markdown("""
                1. **上传** 多张图片（可多选或拖拽上传）
                2. **选择** 检测类别 — 将应用于所有图片
                3. **点击** 开始检测
                4. 查看 **数据看板**：KPI 指标、图表、逐图详情
                """)


if __name__ == "__main__":
    main()

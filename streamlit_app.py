import streamlit as st
import numpy as np
from scipy.signal import fftconvolve
from pydub import AudioSegment
import io
import json
import base64
import os # ファイル存在確認用に追加
import streamlit.components.v1 as components

st.set_page_config(page_title="Universal Audio Reverb (Stable Build)")
st.title("Universal Audio Reverb")

# --- デフォルトIRの設定 ---
DEFAULT_IR_PATH = "default_ir.wav" # ここにデフォルトにしたいファイル名を指定

# 1. 起動時にセッション状態を初期化
if 'ir_base64' not in st.session_state:
    # デフォルトIRファイルが存在する場合、初期値として読み込む
    if os.path.exists(DEFAULT_IR_PATH):
        with open(DEFAULT_IR_PATH, "rb") as f:
            ir_data = f.read()
            st.session_state.ir_base64 = base64.b64encode(ir_data).decode('utf-8')
            st.session_state.ir_name = f"{DEFAULT_IR_PATH} (Default)"
    else:
        st.session_state.ir_base64 = None
        st.session_state.ir_name = None

if "dry_v" not in st.session_state:
    st.session_state["dry_v"] = 1.00
if "wet_v" not in st.session_state:
    st.session_state["wet_v"] = 0.10
if "mast_v" not in st.session_state:
    st.session_state["mast_v"] = 1.00
if "lim_v" not in st.session_state:
    st.session_state["lim_v"] = True
if "result_audio_data" not in st.session_state:
    st.session_state["result_audio_data"] = None

def load_audio(file):
    if isinstance(file, io.BytesIO):
        file.seek(0)
    audio = AudioSegment.from_file(file)
    fs = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    samples /= (1 << (8 * audio.sample_width - 1))
    return fs, samples, audio.channels

# --- サイドバー設定管理 ---
st.sidebar.header("Settings Management")
conf_file = st.sidebar.file_uploader("設定読み込み (.json)", type=["json"])
if conf_file is not None:
    conf_data = json.load(conf_file)
    st.session_state["dry_v"] = conf_data.get("dry_gain", 1.00)
    st.session_state["wet_v"] = conf_data.get("wet_gain", 0.10)
    st.session_state["mast_v"] = conf_data.get("master_gain", 1.00)
    st.session_state["lim_v"] = conf_data.get("use_limiter", True)
    st.session_state.ir_base64 = conf_data.get("ir_base64")
    st.session_state.ir_name = conf_data.get("ir_file_name")

def sync_input(key): st.session_state[key] = st.session_state[key + "_n"]
def sync_slider(key): st.session_state[key + "_n"] = st.session_state[key]

st.sidebar.write("Dry Gain")
dry_gain = st.sidebar.slider("Dry Slider", 0.0, 2.0, key="dry_v", step=0.01, label_visibility="collapsed", on_change=sync_slider, args=("dry_v",))
st.sidebar.number_input("Dry Input", 0.0, 2.0, key="dry_v_n", step=0.01, label_visibility="collapsed", on_change=sync_input, args=("dry_v",), value=st.session_state["dry_v"])
st.sidebar.write("Wet Gain Master")
wet_gain = st.sidebar.slider("Wet Slider", 0.0, 2.0, key="wet_v", step=0.01, label_visibility="collapsed", on_change=sync_slider, args=("wet_v",))
st.sidebar.number_input("Wet Input", 0.0, 2.0, key="wet_v_n", step=0.01, label_visibility="collapsed", on_change=sync_input, args=("wet_v",), value=st.session_state["wet_v"])
st.sidebar.write("Master Output Gain")
master_gain = st.sidebar.slider("Master Slider", 0.0, 2.0, key="mast_v", step=0.01, label_visibility="collapsed", on_change=sync_slider, args=("mast_v",))
st.sidebar.number_input("Master Input", 0.0, 2.0, key="mast_v_n", step=0.01, label_visibility="collapsed", on_change=sync_input, args=("mast_v",), value=st.session_state["mast_v"])
use_limiter = st.sidebar.checkbox("音割れ防止", key="lim_v")

# 2. ファイルアップロード
uploaded_file = st.file_uploader("音源ファイルをアップロード", type=["wav", "mp3", "m4a", "flac"])
ir_file = st.file_uploader("IR(響き)ファイルをアップロード", type=["wav", "mp3", "m4a", "flac"])

# IRの決定ロジック
active_ir_bytes = None
if ir_file is not None:
    # ユーザーが新しくアップロードした場合
    active_ir_bytes = ir_file.getvalue()
    st.session_state.ir_name = ir_file.name
    st.session_state.ir_base64 = base64.b64encode(active_ir_bytes).decode('utf-8')
elif st.session_state.ir_base64 is not None:
    # デフォルトIRまたは設定から読み込まれたIRがある場合
    active_ir_bytes = base64.b64decode(st.session_state.ir_base64)
    st.info(f"使用中のIR: {st.session_state.ir_name}")

if st.session_state.ir_base64 is not None:
    save_dict = {"dry_gain": dry_gain, "wet_gain": wet_gain, "master_gain": master_gain, "use_limiter": use_limiter, "ir_file_name": st.session_state.ir_name, "ir_base64": st.session_state.ir_base64}
    st.sidebar.download_button("設定を保存する", data=json.dumps(save_dict), file_name="reverb_settings.json")

# --- 処理メイン ---
if uploaded_file and active_ir_bytes:
    if st.button("処理を開始"):
        with st.spinner("処理中..."):
            fs, data, channels = load_audio(uploaded_file)
            fs_ir, ir, ir_channels = load_audio(io.BytesIO(active_ir_bytes))
            if channels == 1:
                wet = fftconvolve(data, ir if ir.ndim == 1 else ir[:, 0], mode='full')
            else:
                left = fftconvolve(data[:, 0], ir[:, 0] if ir.ndim > 1 else ir, mode='full')
                right = fftconvolve(data[:, 1], ir[:, 1] if ir.ndim > 1 else ir, mode='full')
                wet = np.vstack((left, right)).T
            dry_padded = np.zeros_like(wet)
            dry_padded[:len(data)] = data
            mixed = (dry_padded * dry_gain) + (wet * wet_gain)
            mixed *= master_gain
            if use_limiter: mixed = np.clip(mixed, -1.0, 1.0)
            
            out_int = (mixed * 32767).astype(np.int16)
            result_bytes = io.BytesIO()
            AudioSegment(out_int.tobytes(), frame_rate=fs, sample_width=2, channels=channels).export(result_bytes, format="wav")
            st.session_state["result_audio_data"] = result_bytes.getvalue()
            st.rerun()

# --- リアルタイム・レベルメーター表示エリア ---
if st.session_state["result_audio_data"] is not None:
    audio_b64 = base64.b64encode(st.session_state["result_audio_data"]).decode()
    
    html_code = f"""
    <div id="meter-container" style="background:#000; padding:15px; border-radius:5px; font-family:sans-serif; color:#ccc;">
        <audio id="audio-player" controls src="data:audio/wav;base64,{audio_b64}" style="width:100%; margin-bottom:15px;"></audio>
        
        <div style="position:relative; height:55px; background:#111; border:1px solid #333; padding:5px;">
            <div style="display:flex; align-items:center; margin-bottom:8px;">
                <span style="width:20px; font-size:10px; color:#888;">L</span>
                <div style="flex-grow:1; position:relative; height:12px; background:#222; overflow:hidden;">
                    <div id="bar-l" style="height:100%; width:0%; background:linear-gradient(90deg, #0f0 75%, #ff0 90%, #f00 100%); transition:width 0.05s;"></div>
                    <div id="peak-l" style="position:absolute; top:0; width:2px; height:100%; background:#fff; display:none; transform:translateX(-1px);"></div>
                </div>
            </div>
            <div style="display:flex; align-items:center; margin-bottom:5px;">
                <span style="width:20px; font-size:10px; color:#888;">R</span>
                <div style="flex-grow:1; position:relative; height:12px; background:#222; overflow:hidden;">
                    <div id="bar-r" style="height:100%; width:0%; background:linear-gradient(90deg, #0f0 75%, #ff0 90%, #f00 100%); transition:width 0.05s;"></div>
                    <div id="peak-r" style="position:absolute; top:0; width:2px; height:100%; background:#fff; display:none; transform:translateX(-1px);"></div>
                </div>
            </div>
            <div style="margin-left:20px; display:flex; justify-content:space-between; font-size:8px; color:#666; border-top:1px solid #444; padding-top:2px;">
                <span>-Inf</span><span>-60</span><span>-40</span><span>-20</span><span>-10</span><span>-6</span><span>-3</span><span>0</span>
            </div>
        </div>
    </div>
    <script>
    const audio = document.getElementById('audio-player');
    const barL = document.getElementById('bar-l');
    const barR = document.getElementById('bar-r');
    const peakL = document.getElementById('peak-l');
    const peakR = document.getElementById('peak-r');
    let audioCtx, source, splitter, anaL, anaR;
    let maxL = 0, maxR = 0;
    audio.onplay = () => {{
        if (!audioCtx) {{
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            source = audioCtx.createMediaElementSource(audio);
            splitter = audioCtx.createChannelSplitter(2);
            anaL = audioCtx.createAnalyser();
            anaR = audioCtx.createAnalyser();
            anaL.fftSize = 256; anaR.fftSize = 256;
            source.connect(splitter);
            splitter.connect(anaL, 0); splitter.connect(anaR, 1);
            source.connect(audioCtx.destination);
            draw();
        }}
    }};
    function getDB(analyser) {{
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteTimeDomainData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {{
            let val = (dataArray[i] - 128) / 128;
            sum += val * val;
        }}
        let rms = Math.sqrt(sum / bufferLength);
        return 20 * Math.log10(rms + 1e-6);
    }}
    function draw() {{
        const dbL = getDB(anaL); const dbR = getDB(anaR);
        const pctL = Math.max(0, (dbL + 60) * (100 / 60));
        const pctR = Math.max(0, (dbR + 60) * (100 / 60));
        barL.style.width = pctL + "%"; barR.style.width = pctR + "%";
        if (pctL > maxL) {{ maxL = pctL; peakL.style.display = 'block'; peakL.style.left = pctL + "%"; }}
        if (pctR > maxR) {{ maxR = pctR; peakR.style.display = 'block'; peakR.style.left = pctR + "%"; }}
        requestAnimationFrame(draw);
    }}
    </script>
    """
    components.html(html_code, height=180)
    st.download_button("保存", st.session_state["result_audio_data"], file_name="output.wav")
from __future__ import annotations

import mimetypes
import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv('API_URL', 'http://localhost:8080')


def call_api(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f'{API_URL}{path}', json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def call_api_file(path: str, file_name: str, file_bytes: bytes, extra_data: dict[str, Any] | None = None) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(file_name)[0] or 'application/octet-stream'
    files = {'file': (file_name, file_bytes, mime_type)}
    response = requests.post(f'{API_URL}{path}', files=files, data=extra_data or {}, timeout=120)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title='KarierAI', layout='wide')
st.title('KarierAI')

with st.sidebar:
    st.markdown(f'**API URL**: `{API_URL}`')
    st.caption('CV upload mendukung PDF teks, PDF scan, dan gambar (PNG/JPG/JPEG/WEBP/BMP/TIFF).')
    if st.button('Reset chat'):
        st.session_state.messages = []

chat_tab, cv_tab, consult_tab = st.tabs(['Chat', 'CV Analyzer', 'Career Consultation'])

with chat_tab:
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input('Tanya tentang lowongan, statistik, atau konsultasi karier')
    if prompt:
        history = '\n'.join(f"{m['role']}: {m['content']}" for m in st.session_state.messages[-20:])
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            try:
                result = call_api('/chat', {'query': prompt, 'history': history})
                st.markdown(result['response'])
                with st.expander('Tool messages'):
                    st.code('\n\n'.join(result.get('tool_messages', [])) or 'No tool messages')
                with st.expander('Usage'):
                    st.json(
                        {
                            'input_tokens': result.get('input_tokens', 0),
                            'output_tokens': result.get('output_tokens', 0),
                            'used_tools': result.get('used_tools', []),
                        }
                    )
                st.session_state.messages.append({'role': 'assistant', 'content': result['response']})
            except Exception as exc:
                st.error(f'Gagal memanggil API: {exc}')

with cv_tab:
    cv_text = st.text_area('Tempel teks CV di sini', height=260)
    cv_file = st.file_uploader('Atau upload CV PDF / gambar', type=['pdf', 'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff'])
    top_k = st.slider('Top K rekomendasi', min_value=1, max_value=10, value=5)
    col1, col2 = st.columns(2)
    if col1.button('Analisis CV'):
        try:
            if cv_file is not None:
                st.json(call_api_file('/cv/analyze-file', cv_file.name, cv_file.getvalue()))
            elif cv_text.strip():
                st.json(call_api('/cv/analyze', {'cv_text': cv_text}))
            else:
                st.warning('Isi teks CV atau upload PDF/gambar terlebih dahulu.')
        except Exception as exc:
            st.error(str(exc))
    if col2.button('Cari rekomendasi kerja'):
        try:
            if cv_file is not None:
                st.json(call_api_file('/recommend-file', cv_file.name, cv_file.getvalue(), {'top_k': top_k}))
            elif cv_text.strip():
                st.json(call_api('/recommend', {'cv_text': cv_text, 'top_k': top_k}))
            else:
                st.warning('Isi teks CV atau upload PDF/gambar terlebih dahulu.')
        except Exception as exc:
            st.error(str(exc))

with consult_tab:
    consult_cv = st.text_area('Teks CV untuk konsultasi karier', height=220, key='consult_cv')
    consult_file = st.file_uploader(
        'Atau upload CV PDF / gambar untuk konsultasi',
        type=['pdf', 'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tif', 'tiff'],
        key='consult_pdf',
    )
    target_role = st.text_input('Target role', value='Data Analyst')
    if st.button('Analisis gap skill'):
        try:
            if consult_file is not None:
                st.json(call_api_file('/consult-file', consult_file.name, consult_file.getvalue(), {'target_role': target_role}))
            elif consult_cv.strip():
                st.json(call_api('/consult', {'cv_text': consult_cv, 'target_role': target_role}))
            else:
                st.warning('Isi teks CV atau upload PDF/gambar terlebih dahulu.')
        except Exception as exc:
            st.error(str(exc))

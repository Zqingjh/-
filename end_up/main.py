# -*- coding: utf-8 -*-
from nicegui import ui
import ollama
import asyncio
import re
import time

from core.config import config
from core.retriever import init_retriever, format_docs
from core.prompt import build_messages
from core.session import (
    init_db, create_session, get_all_sessions, save_message,
    get_messages, update_session_title, delete_session
)

print("正在加载知识库(ChromaDB)...")
retriever = init_retriever()
ollama_client = ollama.AsyncClient(host=config.ollama_host)

class ChatApp:

    def __init__(self):
        self.current_session_id = None
        self.drawer = None
        self.chat_container = None
        self.chat_input = None
        self.generating_states = {}
        self.last_metrics = ""
        self.use_rag = True
        self.temperature = 0.3
        self.top_p = 0.9
        self.max_tokens = 1024
        self.current_model = config.finetuned_model
        self.base_model = config.base_model
        init_db()
        sessions = get_all_sessions()
        self.current_session_id = sessions[0][0] if sessions else create_session()

    def update_input_ui(self):
        if not self.chat_input:
            return
        if self.current_session_id in self.generating_states:
            self.chat_input.disable()
            self.chat_input._props['placeholder'] = 'AI正在思考中...'
        else:
            self.chat_input.enable()
            self.chat_input._props['placeholder'] = f'向{config.assistant_name}提问...'
        self.chat_input.update()

    def switch_session(self, session_id):
        self.current_session_id = session_id
        self.last_metrics = ""
        self.draw_sidebar_content.refresh()
        self.draw_chat_area.refresh()
        self.update_input_ui()

    def new_session(self):
        if len(get_all_sessions()) >= 3:
            ui.notify('对话窗口已达上限(3个)，请先删除一些历史对话',
        type='warning', position='top', timeout=3000)
            return
        self.current_session_id = create_session()
        self.draw_sidebar_content.refresh()
        self.draw_chat_area.refresh()
        self.update_input_ui()

    def delete_session_action(self, session_id):
        delete_session(session_id)
        if self.current_session_id == session_id:
            sessions = get_all_sessions()
            self.current_session_id = sessions[0][0] if sessions else create_session()
            self.draw_chat_area.refresh()
            self.update_input_ui()
        self.draw_sidebar_content.refresh()

    async def send_message(self, e=None):
        if self.current_session_id in self.generating_states:
            ui.notify('辅导员正在处理上一条信息，请稍候', type='warning', position='top')
            return

        user_text = self.chat_input.value
        if not user_text or not user_text.strip():
            return

        self.chat_input.value = ''
        active_session_id = self.current_session_id

        self.generating_states[active_session_id] = {
            'full_response': '',
            'formatted_response': '',
            'markdown_element': None,
            'spinner': None,
            'spinner_visible': True,
            'is_first_chunk': True,
            'ttft': 0.0
        }
        self.update_input_ui()

        try:
            save_message(active_session_id, 'user', user_text)
            history = get_messages(active_session_id)
            if len(history) == 1:
                update_session_title(active_session_id, user_text[:10] + '...')
                self.draw_sidebar_content.refresh()

            self.draw_chat_area.refresh()
            await asyncio.sleep(0.1)

            context_text = ''
            if self.use_rag and retriever:
                docs = retriever.invoke(user_text)
                if docs:
                    context_text = format_docs(docs)

            messages_for_llm = build_messages(user_text, context_text, history)
            state = self.generating_states[active_session_id]
            req_start_time = time.time()

            response = await ollama_client.chat(
                model=self.current_model,
                messages=messages_for_llm,
                stream=True,
                options={
                    'temperature': self.temperature,
                    'top_p': self.top_p,
                    'num_predict': self.max_tokens
                }
            )

            async for chunk in response:
                if state['is_first_chunk']:
                    ttft = time.time() - req_start_time
                    state['ttft'] = ttft
                    with open('metrics_ttft.log', 'a') as f:
                        f.write(f'{time.time()},{ttft:.4f}\n')
                    state['is_first_chunk'] = False
                    state['spinner_visible'] = False
                    if state.get('spinner'):
                        try:
                            state['spinner'].visible = False
                        except Exception:
                            pass

                content = chunk['message'].get('content', '')
                if content:
                    state['full_response'] += content
                    fr = state['full_response']
                    formatted = re.sub(r'([^\n])\s*(\d+\.)\s', r'\1\n\n\2 ', fr)
                    state['formatted_response'] = formatted
                    if (self.current_session_id == active_session_id
                            and state.get('markdown_element')):
                        try:
                            state['markdown_element'].set_content(formatted)
                            ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
                        except Exception:
                            pass
            total_time = time.time() - req_start_time
            ttft_val = state.get('ttft', 0.0)
            metrics_line = (f"模型: {self.current_model} | "
                           f"首字延迟: {ttft_val:.2f}s | 总耗时: {total_time:.2f}s")

            final_response = state['formatted_response']
            if not state['full_response'].strip():
                final_response = '> 警告：模型返回了空白内容。'
            else:
                final_response += f"\n\n---\n*{metrics_line}*"

            if (self.current_session_id == active_session_id
                    and state.get('markdown_element')):
                try:
                    state['markdown_element'].set_content(final_response)
                except Exception:
                    pass

            save_message(active_session_id, 'assistant', final_response)


        except Exception as ex:
            err_msg = f'**模型请求失败**: \n\n`	ext\n{str(ex)}\n`'
            state = self.generating_states.get(active_session_id)
            if state:
                if state.get('spinner'):
                    try:
                        state['spinner'].visible = False
                    except Exception:
                        pass
                if (self.current_session_id == active_session_id
                        and state.get('markdown_element')):
                    try:
                        state['markdown_element'].set_content(err_msg)
                    except Exception:
                        pass
            save_message(active_session_id, 'assistant', err_msg)

        finally:
            if active_session_id in self.generating_states:
                del self.generating_states[active_session_id]
            if self.current_session_id == active_session_id:
                self.update_input_ui()
                try:
                    self.chat_input.run_method('focus')
                except Exception:
                    pass

    @ui.refreshable
    def draw_sidebar_content(self):
        ui.label(config.ui_title).classes('text-xl font-bold mb-4 text-slate-800')
        ui.button('新建对话', on_click=self.new_session).classes(
            'w-full mb-6 bg-blue-500 text-white shadow-sm')

        ui.label('历史对话').classes('text-sm text-slate-400 mb-2 font-semibold')
        with ui.column().classes('w-full gap-1'):
            for s_id, title in get_all_sessions():
                is_active = (s_id == self.current_session_id)
                btn = ui.button(title, on_click=lambda s=s_id: self.switch_session(s))
                if is_active:
                    btn.props('color=primary unelevated')
                else:
                    btn.props('flat color=grey-8')
                btn.classes('w-full text-left justify-start px-3 py-2 rounded-lg')
                with btn:
                    with ui.context_menu():
                        ui.menu_item('删除对话',
                            on_click=lambda s=s_id: self.delete_session_action(s)
                        ).classes('text-red-500')

        ui.separator().classes('my-4')
        ui.label('模型与策略控制').classes('text-sm text-slate-400 mb-4 font-semibold')
        ui.label('模型切换 (对比测试)').classes('text-sm text-slate-500 mb-1 font-semibold')
        ui.toggle({
            config.finetuned_model: '微调模型',
            self.base_model: '基座模型'
        }).bind_value(self, 'current_model').classes('w-full mb-4 text-xs font-bold').props('spread')

        ui.switch('启用RAG').bind_value(self, 'use_rag').classes('w-full font-bold text-slate-700 mb-2')

        with ui.row().classes('w-full items-center justify-between text-slate-600 mt-2'):
            ui.label('温度 (Temp)')
            ui.label().bind_text_from(self, 'temperature', backward=lambda v: f'{v:.2f}')
        ui.slider(min=0.0, max=1.0, step=0.05).bind_value(self, 'temperature').classes('w-full')

        with ui.row().classes('w-full items-center justify-between text-slate-600 mt-2'):
            ui.label('Top-P')
            ui.label().bind_text_from(self, 'top_p', backward=lambda v: f'{v:.2f}')
        ui.slider(min=0.0, max=1.0, step=0.05).bind_value(self, 'top_p').classes('w-full')

        with ui.row().classes('w-full items-center justify-between text-slate-600 mt-2'):
            ui.label('最大生成长度 (Tokens)')
            ui.label().bind_text_from(self, 'max_tokens', backward=lambda v: f'{int(v)}')
        ui.slider(min=128, max=4096, step=128).bind_value(self, 'max_tokens').classes('w-full')

    @ui.refreshable
    def draw_chat_area(self):
        self.chat_container = ui.column().classes('w-full max-w-4xl mx-auto py-8 px-4 flex-grow')
        with self.chat_container:
            messages = get_messages(self.current_session_id)
            if not messages and self.current_session_id not in self.generating_states:
                with ui.column().classes('w-full h-64 items-center justify-center text-slate-300'):
                    ui.icon('school', size='4rem')
                    ui.label(f'同学你好！关于{config.domain_name}规章制度有什么想问的吗？').classes('text-lg mt-2')

            for msg in messages:
                is_user = msg['role'] == 'user'
                with ui.chat_message(
                        name='\u4f60' if is_user else config.assistant_name,
                        avatar=config.avatar_user if is_user else config.avatar_bot,
                        sent=not is_user,
                ):
                    ui.markdown(msg['content'])
            if self.current_session_id in self.generating_states:
                state = self.generating_states[self.current_session_id]
                with ui.chat_message(name=config.assistant_name, avatar=config.avatar_bot, sent=True):
                    spinner = ui.spinner('dots', size='sm')
                    spinner.visible = state.get('spinner_visible', True)
                    state['spinner'] = spinner
                    default_wait_text = '正在检索规章并思考...' if self.use_rag else '正在思考...'
                    content = state.get('formatted_response', '') or default_wait_text
                    state['markdown_element'] = ui.markdown(content)

        ui.run_javascript('setTimeout(() => window.scrollTo(0, document.body.scrollHeight), 100)')


@ui.page('/')
def index():
    ui.query('body').classes('bg-white m-0 p-0 text-slate-800')
    app_instance = ChatApp()

    with ui.left_drawer(value=True).classes(
            'bg-slate-50 border-r border-slate-200 p-4'
    ) as app_instance.drawer:
        app_instance.draw_sidebar_content()

    with ui.column().classes('w-full h-screen no-wrap justify-between bg-white'):
        with ui.scroll_area().classes('w-full flex-grow items-stretch'):
            app_instance.draw_chat_area()

        with ui.row().classes('w-full bg-white border-t border-slate-100 p-4 pb-8'):
            with ui.row().classes('w-full max-w-4xl mx-auto relative items-center'):
                app_instance.chat_input = (
                    ui.input(placeholder=f'向{config.assistant_name}提问...')
                    .classes('w-full text-lg')
                    .props('outlined rounded item-aligned')
                )
                app_instance.chat_input.on('keydown.enter', app_instance.send_message)
                ui.label('按 Enter 键发送').classes('absolute right-4 text-xs text-slate-400')


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(title=config.ui_title, dark=False, port=config.ui_port)

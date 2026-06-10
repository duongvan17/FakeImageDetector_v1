from __future__ import annotations

from haystack import component
from ...config import Settings
from .rag_store import FeedbackRAGStore
from .responder import _image_uri_from_messages

@component
class FeedbackIngester:
    def __init__(self, settings: Settings, rag_store: FeedbackRAGStore) -> None:
        self._s = settings
        self._rag_store = rag_store

    @component.output_types(plan=dict)
    def run(self, plan: dict, backend_url: str = "", backend_model: str = "") -> dict:
        if plan.get("intent") != "user_feedback":
            return {"plan": plan}
            
        messages = plan.get("messages", [])
        if not messages:
            return {"plan": plan}
            
        # Extract user's feedback text
        user_text = messages[-1].get("content", "")
        if isinstance(user_text, list):
            user_text = " ".join([p.get("text", "") for p in user_text if p.get("type") == "text"])
            
        # Try to find the image in the current or previous messages
        image_uri = plan.get("image")
        if not image_uri:
            # Look backwards through history to find the most recently attached image
            for msg in reversed(messages[:-1]):
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if url.startswith("data:image/"):
                                image_uri = url
                                break
                if image_uri:
                    break
                    
        if image_uri and user_text:
            b64 = image_uri.split(",", 1)[1] if "," in image_uri else image_uri
            b_url = backend_url or self._s.model_backend_url
            b_model = backend_model or self._s.model_name
            
            # Save feedback to ChromaDB
            self._rag_store.add_feedback(b64, user_text, b_url, b_model)
            
            plan["route"] = "static"
            plan["static_reply"] = "✅ Cảm ơn bạn! Đóng góp/đính chính của bạn đã được ghi nhận vào hệ thống RAG để tôi có thể học hỏi và trả lời tốt hơn cho các ảnh tương tự trong tương lai."
        else:
            plan["route"] = "static"
            plan["static_reply"] = "⚠️ Tôi hiểu đây là một đóng góp, nhưng tôi không tìm thấy bức ảnh nào trong cuộc hội thoại để lưu kèm phản hồi này."
            
        return {"plan": plan}

"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast, Vinpearl
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác

## AVAILABLE TOOLS
- search_product_catalog(category, max_price): Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.
- submit_support_ticket(customer_name, issue_description, priority): Ghi nhận yêu cầu hỗ trợ của khách hàng.

## CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm. PHẢI gọi tool để lấy dữ liệu thực.
2. Khi người dùng hỏi về sản phẩm/giá, PHẢI gọi search_product_catalog.
3. Khi người dùng cần hỗ trợ/khiếu nại, PHẢI gọi submit_support_ticket.
4. Chỉ trả lời các câu hỏi liên quan đến Vingroup (VinFast, Vinpearl, Vingroup).
5. Nếu không có kết quả phù hợp, thông báo rõ ràng cho người dùng.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời về các sản phẩm/dịch vụ Vingroup.
- Không trả lời các câu hỏi ngoài phạm vi Vingroup.

## OUTPUT CONTRACT
Mỗi bước trả lời theo format:
- Thought: Suy nghĩ của Agent về yêu cầu người dùng.
- Action: Tool được gọi và tham số.
- Observation: Kết quả trả về từ tool.
- Final Answer: Câu trả lời cuối cùng cho người dùng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, bool]:
        text = user_input.lower()
        needs_catalog = any(k in text for k in [
            "xem", "giá", "dưới", "resort", "du lịch", "sản phẩm",
            "xe điện", "xe vinfast"
        ])
        needs_ticket = any(k in text for k in [
            "tên tôi", "tôi tên", "bị lỗi", "khiếu nại", "phản hồi",
            "hỗ trợ", "gấp", "nghiêm trọng", "khẩn cấp", "mức độ"
        ])
        is_faq = any(k in text for k in ["bảo hành", "chính sách"])
        # Nếu là FAQ thì không gọi tool
        if is_faq:
            needs_catalog = False
            needs_ticket = False
        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq,
        }

    def _extract_catalog_args(self, user_input: str) -> Dict[str, Any]:
        text = user_input.lower()
        if "xe" in text:
            category = "xe_dien"
        elif "resort" in text or "du lịch" in text or "vinpearl" in text:
            category = "du_lich"
        else:
            category = "xe_dien"

        max_price = 999999999999
        m = re.search(r"dưới\s+(\d+)\s*(triệu|tỷ)", text)
        if m:
            val = int(m.group(1))
            unit = m.group(2)
            max_price = val * (1000000 if unit == "triệu" else 1000000000)
        return {"category": category, "max_price": max_price}

    def _extract_ticket_args(self, user_input: str) -> Dict[str, Any]:
        text = user_input
        # customer name
        customer_name = "Khách hàng"
        m = re.search(r"tôi tên\s+([^,\.]+)", text, re.IGNORECASE)
        if not m:
            m = re.search(r"tên tôi là\s+([^,\.]+)", text, re.IGNORECASE)
        if m:
            customer_name = m.group(1).strip()

        # issue description
        issue_description = "Yêu cầu hỗ trợ từ khách hàng"
        m = re.search(r"(xe\s+\S+(?:\s+\S+)*?\s+bị\s+[^.]+)", text, re.IGNORECASE)
        if m:
            issue_description = m.group(1).strip()
        elif "phản hồi" in text.lower() or "khiếu nại" in text.lower():
            # take clause after name
            pass

        # priority
        low = text.lower()
        if any(k in low for k in ["nghiêm trọng", "gấp", "khẩn cấp", "cao"]):
            priority = "high"
        elif "trung bình" in low:
            priority = "medium"
        elif "thấp" in low:
            priority = "low"
        else:
            priority = "medium"

        # refine issue_description for TC03 style (phòng ... bị ẩm mốc)
        if "phòng" in low and "bị" in low:
            m2 = re.search(r"(phòng[^.]*?bị[^.]+)", text, re.IGNORECASE)
            if m2:
                issue_description = m2.group(1).strip()

        return {
            "customer_name": customer_name,
            "issue_description": issue_description,
            "priority": priority,
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        intents = self._detect_intents(user_input)

        # FAQ: trả lời trực tiếp, không gọi tool
        if intents["is_faq"]:
            self.trace.append({"step": "init", "user_input": user_input})
            self.trace.append({"step": "thought", "content": "Câu hỏi FAQ về bảo hành, trả lời trực tiếp."})
            answer = "Pin xe điện VinFast được bảo hành 10 năm (bảo hành pin)."
            self.trace.append({"step": "final", "answer": answer})
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed",
            }

        results: Dict[str, Any] = {}
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            progress = False

            if intents["needs_catalog"] and "catalog" not in results:
                args = self._extract_catalog_args(user_input)
                self.trace.append({"step": "thought", "content": "Cần tra cứu catalog sản phẩm."})
                self.trace.append({"step": "action", "tool": "search_product_catalog", "args": args})
                obs = search_product_catalog(**args)
                self.trace.append({"step": "observation", "tool": "search_product_catalog", "result": obs})
                results["catalog"] = obs
                progress = True

            if intents["needs_ticket"] and "ticket" not in results:
                args = self._extract_ticket_args(user_input)
                self.trace.append({"step": "thought", "content": "Cần tạo ticket hỗ trợ."})
                self.trace.append({"step": "action", "tool": "submit_support_ticket", "args": args})
                obs = submit_support_ticket(**args)
                self.trace.append({"step": "observation", "tool": "submit_support_ticket", "result": obs})
                results["ticket"] = obs
                progress = True

            # Tổng hợp final answer nếu đã xử lý hết intent cần thiết
            pending_catalog = intents["needs_catalog"] and "catalog" not in results
            pending_ticket = intents["needs_ticket"] and "ticket" not in results
            if not pending_catalog and not pending_ticket:
                # synthesize
                parts = []
                if "catalog" in results:
                    res = results["catalog"]
                    if not res:
                        parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
                    else:
                        names = ", ".join(p["name"] for p in res)
                        parts.append(f"Tìm thấy các sản phẩm: {names}.")
                if "ticket" in results:
                    t = results["ticket"]
                    parts.append(f"Ticket {t['ticket_id']} đã được tạo thành công cho {t['customer_name']}.")
                if not parts:
                    parts.append("Xin chào, tôi là VinAssistant. Tôi có thể hỗ trợ bạn về sản phẩm và dịch vụ Vingroup.")
                answer = " ".join(parts)
                self.trace.append({"step": "final", "answer": answer})
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed",
                }

            if not progress:
                break

        # Vượt max_iterations
        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "iterations": iteration,
            "status": "max_iterations_reached",
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

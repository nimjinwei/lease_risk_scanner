import streamlit as st
import tempfile
import os
import json
import fitz  # PyMuPDF
import google.generativeai as genai
import anthropic
import io
import base64
from fpdf import FPDF
import datetime

# ====================== PROMPTS ======================
LEASE_EXPERT_PROMPT = """
You are a highly experienced Australian Tenant Rights Expert and Lawyer. Your goal is to analyze residential lease agreements to protect the tenant from unfair clauses, hidden fees, and unusual conditions.

You must output your analysis STRICTLY as a JSON object with the following schema. DO NOT wrap it in markdown code blocks like ```json or anything else, just output the raw JSON string.

{
  "basic_info": {
    "rent": "string (e.g. '$500/week', or 'Not found')",
    "bond": "string (e.g. '$2000', or 'Not found')",
    "lease_term": "string (e.g. '12 months', or 'Not found')"
  },
  "risk_score": "Low | Med | High",
  "risk_clauses": [
    {
      "severity": "High | Med | Low",
      "original_text": "string (brief quote of the problematic clause)",
      "explanation": "string (Chinese explanation of WHY this is risky/unfair according to Australian residential tenancy laws)",
      "suggestion": "string (Chinese suggestion on what to do)"
    }
  ],
  "negotiation_templates": [
    {
      "title": "string (Chinese title of the negotiation, e.g. '要求移除专业清洁条款')",
      "email_text": "string (English email template for the tenant to send to the agent/landlord)"
    }
  ]
}

If no major risks are found, you can set risk_score to "Low" and leave risk_clauses empty or note minor things.
If there are standard but slightly unfair clauses (like mandatory professional cleaning when not keeping pets, which is illegal in some states), flag them as High or Med risk.
"""

BOND_ASSISTANT_PROMPT = """
You are an expert Australian property manager and tenant advocate. Your goal is to review descriptions or photos of a property's condition at the end of a lease and predict potential Bond deductions.

You must output your analysis STRICTLY as a JSON object with the following schema. DO NOT wrap it in markdown code blocks like ```json, just output the raw JSON string.

{
  "overall_risk": "Low | Med | High",
  "potential_deductions": [
    {
      "issue": "string (Chinese short description, e.g. '地毯污渍')",
      "estimated_cost_range": "string (e.g. '$50 - $150')",
      "is_fair_wear_and_tear": true/false,
      "explanation": "string (Chinese explanation of why this might be deducted or why it's fair wear and tear)"
    }
  ],
  "defense_strategy": "string (Chinese advice on how to dispute unfair claims to the tribunal, e.g. VCAT/NCAT)"
}
"""

# ====================== CONFIG & STATE ======================
st.set_page_config(
    page_title="澳洲租约 AI 避坑助手",
    page_icon="🏠",
    layout="wide"
)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "bond_result" not in st.session_state:
    st.session_state.bond_result = None

# ====================== HELPER FUNCTIONS ======================

def extract_pdf_images(uploaded_file):
    """Convert PDF pages to base64 images for Vision APIs (Multimodal Priority)"""
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        images = []
        # Limit to first 15 pages to avoid massive payloads
        for page_num in range(min(15, len(doc))):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_bytes = pix.tobytes("jpeg")
            b64_encoded = base64.b64encode(img_bytes).decode("utf-8")
            images.append({
                "type": "image",
                "media_type": "image/jpeg",
                "data": b64_encoded
            })
        return images
    except Exception as e:
        st.warning(f"PDF 转图片失败，将尝试提取纯文本: {e}")
        return None

def extract_pdf_text(uploaded_file):
    """Fallback: Extract text from PDF"""
    try:
        uploaded_file.seek(0)
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return str(e)

def parse_json_response(raw_text):
    """Safely parse LLM output looking for JSON"""
    try:
        # Sometimes LLMs still wrap in ```json ... ``` despite instructions
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0]
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0]
        
        return json.loads(raw_text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"AI 返回的数据格式无法解析为 JSON: {str(e)}\nRaw output: {raw_text[:200]}...")

def call_gemini(api_key, system_prompt, images, text, model_name="gemini-1.5-pro"):
    genai.configure(api_key=api_key)
    # Gemini 1.5 Pro is highly multimodal
    model = genai.GenerativeModel(model_name)
    
    contents = [system_prompt]
    if images:
        for img in images:
            img_bytes = base64.b64decode(img["data"])
            pil_img = Image.open(io.BytesIO(img_bytes))
            contents.append(pil_img)
    if text:
        contents.append(text)
        
    response = model.generate_content(
        contents,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            # force JSON structure via prompt
        )
    )
    return parse_json_response(response.text)

def call_claude(api_key, system_prompt, images, text, model_name="claude-3-5-sonnet-20240620"):
    client = anthropic.Anthropic(api_key=api_key)
    
    content_blocks = []
    if images:
        for img in images:
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img["media_type"],
                    "data": img["data"]
                }
            })
    if text:
        content_blocks.append({
            "type": "text",
            "text": text
        })
        
    # Claude requires system prompt separated
    response = client.messages.create(
        model=model_name,
        max_tokens=4000,
        temperature=0.1,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": content_blocks
            }
        ]
    )
    return parse_json_response(response.content[0].text)

# PDF Generation
class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        font_path = "SimHei.ttf"
        if not os.path.exists(font_path):
            import urllib.request
            try:
                urllib.request.urlretrieve('https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf', font_path)
            except Exception:
                pass
        
        if os.path.exists(font_path):
            self.add_font("SimHei", "", font_path)
            self.default_font = "SimHei"
        else:
            self.default_font = "Arial"

    def header(self):
        self.set_font(self.default_font, '', 15)
        self.cell(0, 10, 'Lease Risk Analysis Report / 租约风险分析报告', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font(self.default_font, '', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font(self.default_font, '', 11)
        if self.default_font == "Arial":
            body = body.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 10, body)
        self.ln()

def generate_pdf_report(data):
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.chapter_title('Basic Information')
    info = data.get('basic_info', {})
    pdf.chapter_body(f"Rent: {info.get('rent', 'N/A')}\nBond: {info.get('bond', 'N/A')}\nLease Term: {info.get('lease_term', 'N/A')}")
    
    pdf.chapter_title(f"Overall Risk Score: {data.get('risk_score', 'Unknown')}")
    
    pdf.chapter_title('Risk Clauses Detected')
    for idx, clause in enumerate(data.get('risk_clauses', [])):
        body = f"[{clause.get('severity', 'N/A')}] Original Text:\n{clause.get('original_text', '')}\n\n"
        body += f"Explanation:\n{clause.get('explanation', '')}\n\n"
        body += f"Suggestion:\n{clause.get('suggestion', '')}"
        pdf.chapter_body(body)
        pdf.ln(2)
        
    # Return as bytes
    return pdf.output(dest='S').encode('latin-1')

# ====================== UI SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ 配置")
    api_provider = st.radio("选择 AI 模型", ["Claude 3.5 Sonnet", "Gemini 3.5 flash"])
    
    api_key_env = st.secrets.get("CLAUDE_API_KEY") if "Claude" in api_provider else st.secrets.get("GEMINI_API_KEY")
    api_key = st.text_input("输入对应的 API Key", value=api_key_env if api_key_env else "", type="password")
    
    st.markdown("---")
    st.markdown("💡 **提示**：直接上传 PDF 或图片，AI 会自动识别文字和条款。多模态优先！")

# ====================== MAIN APP ======================
st.title("🏠 澳洲租约 AI 避坑助手")
st.markdown("**全自动化租约扫描，揪出霸王条款，守护你的 Bond！** (Multimodal LLM Powered)")

tab1, tab2, tab3 = st.tabs(["📄 租约分析", "📊 风险仪表盘", "💰 Bond 助手"])

with tab1:
    st.subheader("1. 上传租约文件")
    uploaded_files = st.file_uploader(
        "支持 PDF、图片（JPG/PNG）", 
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🚀 开始 AI 分析", type="primary", use_container_width=True)
    with col2:
        demo_btn = st.button("👀 使用演示租约", use_container_width=True)
        
    if demo_btn:
        st.session_state.analysis_result = {
            "basic_info": {"rent": "$650/week", "bond": "$2600", "lease_term": "12 months"},
            "risk_score": "High",
            "risk_clauses": [
                {
                    "severity": "High",
                    "original_text": "The tenant must arrange and pay for professional carpet cleaning at the end of the tenancy regardless of the condition.",
                    "explanation": "在维州和新州，除非租客养宠物，否则房东不能强制要求退租时必须进行'专业清洁'（Professional Cleaning）。这属于霸王条款。",
                    "suggestion": "在签合同前发邮件要求删除此条款，或者签了之后在退租时以当地 Residential Tenancies Act 为由拒绝支付。"
                },
                {
                    "severity": "Med",
                    "original_text": "Rent increases by 10% every 6 months.",
                    "explanation": "通常固定期限租约内，除非合同明确写明了租金上涨的具体金额或计算方式，否则不能随意涨租。如果是 rolling lease，涨租频率也有法定限制（通常是12个月一次）。",
                    "suggestion": "仔细核对当地州法对涨租频率的限制，并要求中介确认涨租是否符合规定。"
                }
            ],
            "negotiation_templates": [
                {
                    "title": "要求移除强制专业清洁条款",
                    "email_text": "Dear Agent,\n\nI noted a clause in the agreement requiring professional carpet cleaning upon vacating. As per the Residential Tenancies Act, standard terms cannot require professional cleaning unless a pet was kept on the premises. Could we please have this clause amended or removed?\n\nThank you."
                }
            ]
        }
        st.success("演示数据加载成功！请点击上方【📊 风险仪表盘】查看。")

    if analyze_btn:
        if not uploaded_files:
            st.error("❌ 请先上传文件")
        elif not api_key:
            st.error("❌ 请在左侧输入 API Key")
        else:
            with st.spinner("AI 正在使用多模态视觉能力扫描你的租约..."):
                try:
                    images = []
                    text_content = ""
                    
                    for f in uploaded_files:
                        if f.name.lower().endswith(".pdf"):
                            extracted_imgs = extract_pdf_images(f)
                            if extracted_imgs:
                                images.extend(extracted_imgs)
                            else:
                                text_content += extract_pdf_text(f) + "\n"
                        else:
                            # Handle images
                            img_bytes = f.read()
                            b64 = base64.b64encode(img_bytes).decode("utf-8")
                            mime = "image/jpeg" if f.name.lower().endswith((".jpg", ".jpeg")) else "image/png"
                            images.append({"type": "image", "media_type": mime, "data": b64})
                    
                    if "Claude" in api_provider:
                        result = call_claude(api_key, LEASE_EXPERT_PROMPT, images, text_content)
                    else:
                        result = call_gemini(api_key, LEASE_EXPERT_PROMPT, images, text_content)
                        
                    st.session_state.analysis_result = result
                    st.success("✅ 分析完成！请点击上方【📊 风险仪表盘】查看结果。")
                    
                except Exception as e:
                    st.error(f"❌ 分析出错: {str(e)}")


with tab2:
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        
        # Risk Score Header
        score = res.get("risk_score", "Unknown")
        color = "green" if score == "Low" else "orange" if score == "Med" else "red"
        st.markdown(f"<h2 style='text-align: center; color: {color};'>综合风险评级: {score}</h2>", unsafe_allow_html=True)
        
        st.divider()
        
        # Basic Info
        st.subheader("📋 租约基础信息")
        c1, c2, c3 = st.columns(3)
        info = res.get("basic_info", {})
        c1.metric("周租金 (Rent)", info.get("rent", "未知"))
        c2.metric("押金 (Bond)", info.get("bond", "未知"))
        c3.metric("租期 (Lease Term)", info.get("lease_term", "未知"))
        
        st.divider()
        
        # Risk Clauses
        st.subheader("⚠️ 风险条款清单")
        clauses = res.get("risk_clauses", [])
        if not clauses:
            st.success("🎉 太棒了！没有检测到明显的风险条款。")
        else:
            for i, clause in enumerate(clauses):
                sev = clause.get("severity", "Unknown")
                border_color = "red" if sev == "High" else "orange" if sev == "Med" else "blue"
                
                with st.expander(f"{'🚨' if sev == 'High' else '⚡'} [风险: {sev}] 条款详情 #{i+1}", expanded=(sev=="High")):
                    st.markdown(f"**📄 原始条款片段：**\n> {clause.get('original_text', '')}")
                    st.markdown(f"**🧠 坑点解析：**\n{clause.get('explanation', '')}")
                    st.markdown(f"**🛡️ 应对建议：**\n{clause.get('suggestion', '')}")
        
        st.divider()
        
        # Templates
        st.subheader("✉️ 谈判话术模板")
        templates = res.get("negotiation_templates", [])
        for t in templates:
            st.markdown(f"**{t.get('title', '模板')}**")
            st.code(t.get('email_text', ''), language="markdown")
            
        st.divider()
        
        # Export PDF
        st.subheader("📄 导出报告")
        try:
            pdf_bytes = generate_pdf_report(res)
            st.download_button(
                label="📥 下载 PDF 分析报告",
                data=pdf_bytes,
                file_name=f"Lease_Report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.warning("⚠️ 导出PDF组件遇到不支持的中文字体。您可以直接截图保存网页内容。")
            
    else:
        st.info("👈 请先在【📄 租约分析】上传文件并执行分析。")


with tab3:
    st.subheader("📸 退租 Bond 扣款预测")
    st.markdown("拍下房屋磨损或破损处的照片，AI 告诉你中介能扣多少钱，以及如何用“Fair Wear and Tear”反驳。")
    
    bond_files = st.file_uploader(
        "上传房屋状况图片 (或输入描述)", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="bond_uploader"
    )
    
    bond_desc = st.text_area("房屋状况描述（例如：客厅地毯有2厘米咖啡渍，墙上有个钉子眼）")
    
    if st.button("🛡️ 评估扣款风险", type="primary"):
        if not bond_files and not bond_desc:
            st.error("请至少上传一张图片或输入一段描述")
        elif not api_key:
            st.error("请输入 API Key")
        else:
            with st.spinner("AI 正在评估你的房屋状况..."):
                try:
                    images = []
                    for f in bond_files:
                        img_bytes = f.read()
                        b64 = base64.b64encode(img_bytes).decode("utf-8")
                        mime = "image/jpeg" if f.name.lower().endswith((".jpg", ".jpeg")) else "image/png"
                        images.append({"type": "image", "media_type": mime, "data": b64})
                    
                    if "Claude" in api_provider:
                        result = call_claude(api_key, BOND_ASSISTANT_PROMPT, images, bond_desc)
                    else:
                        result = call_gemini(api_key, BOND_ASSISTANT_PROMPT, images, bond_desc)
                        
                    st.session_state.bond_result = result
                    st.success("✅ 评估完成！")
                except Exception as e:
                    st.error(f"❌ 评估出错: {str(e)}")
                    
    if st.session_state.bond_result:
        b_res = st.session_state.bond_result
        st.divider()
        b_score = b_res.get("overall_risk", "Unknown")
        b_color = "green" if b_score == "Low" else "orange" if b_score == "Med" else "red"
        st.markdown(f"**总体扣款风险:** <span style='color: {b_color}; font-size: 20px; font-weight: bold;'>{b_score}</span>", unsafe_allow_html=True)
        
        st.markdown("### 🔍 预估扣款项")
        for item in b_res.get("potential_deductions", []):
            is_fair = item.get('is_fair_wear_and_tear', False)
            tag = "✅ 合理折旧 (免责)" if is_fair else "⚠️ 可能扣款"
            with st.expander(f"{item.get('issue', '未知问题')} | 估算: {item.get('estimated_cost_range', 'N/A')} | {tag}"):
                st.write(item.get('explanation', ''))
                
        st.markdown("### 🥊 维权/抗辩策略")
        st.info(b_res.get("defense_strategy", "收集入住时的 Condition Report 照片，并主张合理折旧。"))
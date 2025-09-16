import gradio as gr
import sys
from pathlib import Path
import json
from datetime import datetime

# Ensure project root is on sys.path
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai_research_agent.src.core.agent.orchestrator import run_pipeline

# Chat history storage
chat_history = []

def format_response(query, response):
    """Format the response in a ChatGPT-like style."""
    formatted = f"🔍 **Recherche effectuée :** {query}\n\n"
    formatted += f"📊 **Détails :** Langue détectée: {response['query_language']} | Intent: {response['intent']} | API: {response['api_used']}\n\n"
    
    if response['results']:
        formatted += f"📚 **Voici les {len(response['results'])} meilleurs articles trouvés :**\n\n"
        
        for i, result in enumerate(response['results'], 1):
            formatted += f"**{i}. {result['title']}**\n"
            formatted += f"👥 Auteurs: {', '.join(result['authors'][:3])}{'...' if len(result['authors']) > 3 else ''}\n"
            formatted += f"📅 Année: {result['year']} | ⭐ Score: {result['score']:.3f}\n"
            formatted += f"🔗 [Lien vers l'article]({result['link']})\n\n"
            
            # Summary
            formatted += f"📝 **Résumé :**\n{result['abstractive_summary']}\n\n"
            formatted += "---\n\n"
    else:
        formatted += "❌ Aucun résultat trouvé pour cette requête.\n"
    
    return formatted

def chat_with_agent(message, history, api_choice, use_pdfs):
    """Handle chat interaction with the research agent."""
    if not message.strip():
        return history, ""
    
    # Add user message to history
    history.append([message, None])
    
    try:
        # Run the pipeline
        with gr.Progress() as progress:
            progress(0.1, desc="Analyse de la requête...")
            response = run_pipeline(
                query=message,
                top_k=3,
                max_pdfs=3,
                api=api_choice,
                use_pdfs=use_pdfs
            )
            progress(0.8, desc="Génération de la réponse...")
        
        # Format the response
        formatted_response = format_response(message, response)
        
        # Update history
        history[-1][1] = formatted_response
        
        return history, ""
        
    except Exception as e:
        error_msg = f"❌ Erreur lors du traitement : {str(e)}"
        history[-1][1] = error_msg
        return history, ""

def clear_chat():
    """Clear the chat history."""
    return [], ""

# Custom CSS for ChatGPT-like styling
css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.chat-message {
    padding: 16px;
    margin: 8px 0;
    border-radius: 12px;
}
.user-message {
    background-color: #f0f0f0;
    margin-left: 20%;
}
.bot-message {
    background-color: #e3f2fd;
    margin-right: 20%;
}
"""

# Create the Gradio interface
with gr.Blocks(css=css, title="AI Research Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Assistant de Recherche Académique IA
    
    Posez vos questions de recherche et obtenez des résumés d'articles scientifiques pertinents !
    
    **Exemples de questions :**
    - "État de l'art en apprentissage profond pour le diagnostic médical"
    - "Recent advances in natural language processing"
    - "Machine learning applications in healthcare"
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                height=600,
                show_label=True,
                container=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Tapez votre question de recherche ici...",
                    label="",
                    lines=2,
                    scale=4
                )
                send_btn = gr.Button("🚀 Rechercher", scale=1, variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Paramètres")
            
            api_choice = gr.Dropdown(
                choices=["arxiv", "semantic_scholar", "pubmed", "crossref"],
                value="arxiv",
                label="🔍 Source de données",
                info="Choisissez l'API de recherche"
            )
            
            use_pdfs = gr.Checkbox(
                label="📄 Traitement PDF complet",
                value=False,
                info="Plus lent mais plus détaillé"
            )
            
            clear_btn = gr.Button("🗑️ Effacer l'historique", variant="secondary")
            
            gr.Markdown("""
            ### 💡 Conseils
            - Utilisez des mots-clés spécifiques
            - Posez des questions en français ou anglais
            - Décochez "Traitement PDF" pour des résultats plus rapides
            """)
    
    # Event handlers
    msg.submit(
        chat_with_agent,
        inputs=[msg, chatbot, api_choice, use_pdfs],
        outputs=[chatbot, msg]
    )
    
    send_btn.click(
        chat_with_agent,
        inputs=[msg, chatbot, api_choice, use_pdfs],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


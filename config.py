



# Style adaptatif pour le mode clair/sombre
style ="""
    <style>
        /* Style adaptatif */
        [data-testid="stAppViewContainer"] {
            background: var(--background-color);
        }
        
        .main-header {
            background: linear-gradient(45deg, var(--primary-color) 30%, var(--secondary-color));
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .feature-container {
            background-color: var(--feature-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: transform 0.2s;
        }
        
        .feature-container:hover {
            transform: translateY(-5px);
        }
        
        .info-box {
            background-color: var(--info-bg);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border: 1px solid var(--border-color);
        }
        
        /* Variables couleurs cadre */
        [data-testid="stAppViewContainer"] {
            --primary-color: #832F37;
            --secondary-color: #9A4434;
            --feature-bg: rgba(255, 255, 255, 0.05);
            --info-bg: rgba(255, 255, 255, 0.05);
            --border-color: rgba(255, 255, 255, 0.1);
            --text-color: inherit;
        }
        
        /* Ajustements pour les conteneurs */
        .stMetric {
            background-color: var(--feature-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }
        
        /* Style pour l'image */
        .wine-image-container {
            display: flex;
            justify-content: center;
            margin: 2rem 0;
        }
        .wine-image-container img {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 100%;
            height: auto;
        }
    </style>
    """
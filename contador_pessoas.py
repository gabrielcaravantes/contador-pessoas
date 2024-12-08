import cv2
import numpy as np
import os
from pathlib import Path

# Obtém o diretório do script atual
SCRIPT_DIR = Path(__file__).parent.absolute()

def baixar_classificadores():
    """Baixa os classificadores necessários"""
    classificadores = {
        'frontal_face': 'haarcascade_frontalface_default.xml',
        'frontal_face_alt': 'haarcascade_frontalface_alt.xml',
        'profile_face': 'haarcascade_profileface.xml'
    }
    
    for nome, arquivo in classificadores.items():
        cascade_path = SCRIPT_DIR / arquivo
        if not cascade_path.exists():
            print(f"Baixando classificador {nome}...")
            url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{arquivo}"
            import urllib.request
            urllib.request.urlretrieve(url, str(cascade_path))

def contar_pessoas(caminho_imagem):
    # Baixa os classificadores se necessário
    baixar_classificadores()
    
    # Carrega os classificadores
    face_cascade = cv2.CascadeClassifier(str(SCRIPT_DIR / 'haarcascade_frontalface_default.xml'))
    face_alt_cascade = cv2.CascadeClassifier(str(SCRIPT_DIR / 'haarcascade_frontalface_alt.xml'))
    profile_cascade = cv2.CascadeClassifier(str(SCRIPT_DIR / 'haarcascade_profileface.xml'))
    
    # Lê a imagem
    imagem = cv2.imread(str(caminho_imagem))
    if imagem is None:
        print(f"Erro ao carregar a imagem: {caminho_imagem}")
        return 0
    
    # Obtém as dimensões da imagem
    altura, largura = imagem.shape[:2]
    
    # Converte para escala de cinza
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Equaliza o histograma para melhorar o contraste
    gray = cv2.equalizeHist(gray)
    
    # Parâmetros ajustados para melhor detecção
    scale_factor = 1.1
    min_neighbors = 4
    min_size = (int(altura/20), int(altura/20))  # Tamanho mínimo proporcional à imagem
    
    # Detecta faces com diferentes classificadores
    faces1 = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
    faces2 = face_alt_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
    
    # Detecta faces de perfil
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
    # Detecta faces de perfil na imagem espelhada
    profiles_flipped = profile_cascade.detectMultiScale(cv2.flip(gray, 1), scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
    
    # Combina todas as detecções
    todas_deteccoes = []
    
    # Adiciona faces frontais
    for faces in [faces1, faces2]:
        for (x, y, w, h) in faces:
            todas_deteccoes.append((x, y, w, h))
    
    # Adiciona faces de perfil
    for (x, y, w, h) in profiles:
        todas_deteccoes.append((x, y, w, h))
    
    # Adiciona faces de perfil da imagem espelhada
    for (x, y, w, h) in profiles_flipped:
        x = largura - x - w  # Ajusta a coordenada x para a imagem original
        todas_deteccoes.append((x, y, w, h))
    
    # Remove detecções sobrepostas com um algoritmo mais preciso
    deteccoes_finais = []
    for (x1, y1, w1, h1) in todas_deteccoes:
        area1 = w1 * h1
        sobreposto = False
        
        # Ignora detecções muito pequenas ou muito grandes
        if area1 < (altura * largura) / 400 or area1 > (altura * largura) / 4:
            continue
            
        for (x2, y2, w2, h2) in deteccoes_finais:
            # Calcula a área de sobreposição
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y
            
            # Se houver sobreposição significativa, mantém apenas a maior detecção
            if overlap_area > 0.3 * min(w1 * h1, w2 * h2):
                sobreposto = True
                if area1 > w2 * h2:
                    # Substitui a detecção anterior pela atual
                    deteccoes_finais.remove((x2, y2, w2, h2))
                    sobreposto = False
                break
                
        if not sobreposto:
            deteccoes_finais.append((x1, y1, w1, h1))
    
    # Desenha retângulos ao redor das pessoas detectadas
    for (x, y, w, h) in deteccoes_finais:
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Adiciona um pequeno círculo no centro da detecção para visualização
        centro_x = x + w//2
        centro_y = y + h//2
        cv2.circle(imagem, (centro_x, centro_y), 2, (0, 0, 255), 2)
    
    # Conta e mostra o número de pessoas
    num_pessoas = len(deteccoes_finais)
    cv2.putText(imagem, f'Pessoas: {num_pessoas}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Cria pasta para salvar resultados se não existir
    pasta_resultados = SCRIPT_DIR / 'resultados'
    pasta_resultados.mkdir(exist_ok=True)
    
    # Salva a imagem com as detecções
    nome_arquivo = f"resultado_{caminho_imagem.name}"
    caminho_resultado = pasta_resultados / nome_arquivo
    cv2.imwrite(str(caminho_resultado), imagem)
    
    return num_pessoas

def processar_pasta_imagens():
    # Define os caminhos das pastas
    pasta_imagens = SCRIPT_DIR / 'imagens'
    
    # Verifica se a pasta de imagens existe
    if not pasta_imagens.exists():
        pasta_imagens.mkdir(exist_ok=True)
        print(f"\nPasta 'imagens' criada em: {pasta_imagens}")
        print("Por favor, coloque algumas imagens na pasta e execute o script novamente.")
        return
    
    # Lista todas as imagens na pasta
    extensoes_imagem = ['.jpg', '.jpeg', '.png', '.bmp']
    imagens = [f for f in pasta_imagens.glob('*') if f.suffix.lower() in extensoes_imagem]
    
    if not imagens:
        print(f"\nNenhuma imagem encontrada na pasta {pasta_imagens}")
        print("Por favor, adicione algumas imagens e execute o script novamente.")
        return
    
    print(f"\nAnalisando {len(imagens)} imagens...")
    print("-" * 50)
    
    # Processa cada imagem
    for imagem_path in imagens:
        print(f"\nProcessando: {imagem_path.name}")
        try:
            total_pessoas = contar_pessoas(imagem_path)
            print(f"Total de pessoas detectadas: {total_pessoas}")
        except Exception as e:
            print(f"Erro ao processar a imagem {imagem_path.name}: {e}")
    
    print("\nProcessamento concluído!")
    print(f"As imagens com as detecções foram salvas na pasta 'resultados'")

if __name__ == "__main__":
    processar_pasta_imagens()

# Contador de Pessoas usando Visão Computacional

Este é um exemplo simples de um sistema que utiliza visão computacional para contar o número de pessoas em imagens.

## Requisitos

- Python 3.6 ou superior
- OpenCV (cv2)
- NumPy

## Instalação

1. Instale as dependências necessárias:
```bash
pip install opencv-python numpy
```

## Como usar

1. Coloque suas imagens na pasta `imagens`
2. Execute o script `contador_pessoas.py`
3. O programa irá:
   - Processar todas as imagens na pasta `imagens`
   - Criar uma pasta `resultados` com as imagens processadas
   - Mostrar no console o número de pessoas detectadas em cada imagem
   - Salvar as imagens processadas com retângulos verdes ao redor das pessoas detectadas

## Estrutura de pastas

```
contador-pessoas/
├── contador_pessoas.py    # Script principal
├── imagens/              # Coloque suas imagens aqui
└── resultados/           # As imagens processadas serão salvas aqui
```

## Formatos de imagem suportados

- JPG/JPEG
- PNG
- BMP

## Limitações

- O detector funciona melhor com pessoas em pé e completamente visíveis
- O desempenho pode variar dependendo da qualidade da imagem e da posição das pessoas
- Algumas pessoas podem não ser detectadas se estiverem parcialmente ocultas ou em posições não convencionais

## Como funciona

O sistema utiliza classificadores Haar Cascade do OpenCV para detectar faces nas imagens. São utilizados três classificadores diferentes:
- Detector frontal de face padrão
- Detector frontal de face alternativo
- Detector de face de perfil

O sistema combina os resultados destes classificadores e aplica um algoritmo de remoção de detecções sobrepostas para obter o resultado final. As faces detectadas são marcadas com retângulos verdes na imagem.

#importando as bibliotecas e lembrando de ter a s3prl
#pip install s3prl

import torch
import torchaudio
import os

def main():

 waveform, sample_rate = torchaudio.load('caminho do dataset')
#pre-preocessamento dos audios, verificar se precisa
 waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
 waveform = torchaudio.transforms.Vad(sample_rate=16000)(waveform)

#carregando o modelo univnet
modelo = torch.hub.load('s3prl/s3prl', 'univnet_tts_libritts')

#convertendo os audios em embaddings usando o modelo 
with torch.no_grad():
    embeddings = modelo.extract_embeddings(waveform)

#criando uma pasta para armazenar 
embeddings_dir = 'caminho/para/a/pasta/embeddings'

if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)


#salvando os vetores de embeddings em arquivos separados na pasta
for i, embedding in enumerate(embeddings):
        embedding_path = os.path.join(embeddings_dir, f'embedding_{i}.pt')
        torch.save(embedding, embedding_path)

if __name__ == "__main__":
    main()
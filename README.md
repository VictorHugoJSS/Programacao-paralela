# Base64 Paralelo: CPU vs GPU

Base64 converte dados binários em texto ASCII agrupando bytes em blocos de 3 e mapeando cada bloco para 4 caracteres de uma tabela de 64 símbolos. A versão sequencial percorre esses blocos um a um, e isso vira gargalo na casa dos megabytes. O ponto-chave é que cada bloco é independentes. O ponto-chave é que cada bloco é independente: nenhum precisa do resultado de outro. Isso torna o problema diretamente paralelizável na GPU.

## Implementação

O projeto tem quatro funções: `base64Encoder` e `base64Decoder` rodam em CPU, em loop sequencial; `base64EncoderGPU` e `base64DecoderGPU` rodam na GPU, com cada thread processando um bloco independente (3 bytes no encoder, 4 caracteres no decoder).

As versões GPU usam `@cuda.jit` do Numba para compilar kernels CUDA direto do Python. O fluxo é converter a string em array NumPy, enviar à GPU com `cuda.to_device`, executar o kernel em paralelo e recuperar o resultado com `copy_to_host`. O padding (`=`) é tratado fora do kernel, na CPU, porque afeta no máximo os 2 bytes finais.

## Resultados

Testes no Google Colab com GPU T4, entradas geradas aleatoriamente:

| Tamanho | CPU Enc (s) | CPU Dec (s) | GPU Enc (s) | GPU Dec (s) |
|---------|------------|------------|------------|------------|
| 1K      | 0.0003     | 0.0003     | 0.0020     | 0.0019     |
| 10K     | 0.0037     | 0.0025     | 0.0009     | 0.0007     |
| 100K    | 0.0311     | 0.0239     | 0.0012     | 0.0010     |
| 1M      | 0.3130     | 0.2531     | 0.0067     | 0.0037     |
| 10M     | 2.9807     | 3.7081     | 0.0655     | 0.0286     |

A GPU começa a vencer a partir de ~10K caracteres. Abaixo disso, o overhead de transferência entre CPU e GPU supera o ganho do paralelismo. Com 10M caracteres, o speedup chega a 45x no encoder e 129x no decoder.

## Limitações

Para entradas pequenas, a CPU ganha. Mover dados para a GPU e de volta não se paga com poucos blocos, o que explica os `NumbaPerformanceWarning` para 1K e 10K. A conversão de strings para arrays NumPy antes de cada chamada ao kernel também adiciona custo, mas é necessária pela interface do Numba.

## Aplicações práticas

Qualquer pipeline que processe Base64 em volume sai ganhando: servidores de upload, sistemas de backup que convertem binários antes de armazenar, pipelines de ML que trafegam dados entre serviços via JSON.

## Como executar

O projeto roda no Google Colab com runtime GPU (T4).

1. Abra o notebook no Colab
2. Vá em **Runtime → Change runtime type → T4 GPU
3. Execute as células em ordem

**Dependências:** Python 3.12, Numba, NumPy, Matplotlib

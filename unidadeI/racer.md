# DeepRacer - AWS

## 🚗 AWS DeepRacer


<img src="https://hostingjournalist.com/wp-content/uploads/2020/01/AWS-DeepRacer-League-Preview.jpg" width="200" height="200" />

### 📚 O que é o AWS DeepRacer?

O AWS DeepRacer é um carro de corrida autônomo em escala 1/18  treinado com redes neurais utilizando aprendizado de reforço (Reinforcement Learning - RL). O AWS DeepRacer foi projetado para fornecer uma maneira divertida e interessante de começar a usar o aprendizado de reforço (RL). 

O AWS DeepRacer é um veículo de código aberto, com uma comunidade de desenvolvedores ativa e crescente, que permite que você experimente e compartilhe novas ideias e recursos.

### 📚 O que é o aprendizado de reforço?

O aprendizado de reforço é uma técnica de aprendizado de máquina que ensina um agente a aprender o comportamento ideal em um ambiente executando ações e vendo os resultados. O agente aprende a alcançar uma meta em um ambiente incerto, onde ele não é informado qual ação executar, mas é recompensado ou punido por suas ações. O agente deve, portanto, determinar por si mesmo qual ação executar para maximizar a recompensa ao longo do tempo.

### 📚 Como funciona o AWS DeepRacer?

Você primeiro criar sua conta clicando em [AWS DeepRacer](https://student.deepracer.com/home) e depois de criar sua conta, você pode acessar o console do AWS DeepRacer e começar a treinar seu modelo de aprendizado de reforço.

### Como treinar seu modelo de aprendizado de reforço?

Para treinar seu modelo de aprendizado de reforço, você precisa criar um modelo de aprendizado de reforço, criar uma pista de corrida, treinar seu modelo e avaliar seu modelo.

#### Criando um modelo de aprendizado de reforço

Para criar um modelo de aprendizado de reforço, você precisa acessar o console do AWS DeepRacer e clicar em **Create model**.

Você precisará definir uma função de recompensa, um algoritmo de aprendizado de reforço e um ambiente de simulação.


# Exemplo de função de recompensa

```python
def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''

    # Read input parameters
    track_width = params['track_width'] 
    distance_from_center = params['distance_from_center']

    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0 # maximum reward
    elif distance_from_center <= marker_2:
        reward = 0.5 # intermediate reward
    elif distance_from_center <= marker_3:
        reward = 0.1 # intermediate reward
    else:
        reward = 1e-3  # likely crashed/ close to off track. We penalize.

    return float(reward)
```

## Video de simulação 

![](https://github.com/roscibely/neural_networks/blob/develop/unidadeI/animation.gif)

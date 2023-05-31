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


## Permaneça na faixa

Neste exemplo, damos uma alta recompensa quando o carro permanece na pista e penalizamos se o carro se desvia dos limites da pista.

Este exemplo usa os parâmetros all_wheels_on_track, distance_from_center e track_width para determinar se o carro está na pista e dar uma alta recompensa em caso afirmativo.

Como esta função não recompensa nenhum tipo específico de comportamento além de permanecer na pista, um agente treinado com esta função pode levar mais tempo para convergir para um determinado comportamento.

```python

def reward_function(params):
    '''
    Example of rewarding the agent to stay inside the two borders of the track
    '''

    # Read input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']

    # Give a very low reward by default
    reward = 1e-3

    # Give a high reward if no wheels go off the track and
    # the agent is somewhere in between the track borders
    if all_wheels_on_track and (0.5*track_width - distance_from_center) >= 0.05:
        reward = 1.0

    # Always return a float value
    return float(reward)

```

## Siga a Linha Central

Neste exemplo, medimos a que distância o carro está do centro da pista e damos uma recompensa maior se o carro estiver próximo da linha central.

Este exemplo usa os parâmetros track_width e distance_from_center e retorna uma recompensa decrescente quanto mais longe o carro estiver do centro da pista.

Este exemplo é mais específico sobre que tipo de comportamento de direção recompensar, portanto, um agente treinado com essa função provavelmente aprenderá a seguir a pista muito bem. No entanto, é improvável que aprenda qualquer outro comportamento, como acelerar ou frear nas curvas.

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

## Evitar zigue-zague

Este exemplo incentiva o agente a seguir a linha central, mas penaliza com uma recompensa menor se virar demais, o que ajudará a evitar o comportamento em zigue-zague.

O agente aprenderá a dirigir suavemente no simulador e provavelmente exibirá o mesmo comportamento quando implantado no veículo físico.

```python
def reward_function(params):
    '''
    Example of penalize steering, which helps mitigate zig-zag behaviors
    '''
    # Read input parameters
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    abs_steering = abs(params['steering_angle']) # Only need the absolute steering angle
    # Calculate 3 marks that are farther and father away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track
    # Steering penality threshold, change the number based on your action space setting
    ABS_STEERING_THRESHOLD = 15 
    # Penalize reward if the car is steering too much
    if abs_steering > ABS_STEERING_THRESHOLD:
        reward *= 0.8
    return float(reward)
```

## Video ambiente de simulação 

![](https://github.com/roscibely/neural_networks/blob/develop/unidadeI/animation.gif)

[Conhecendo AWS DeepRacer races](https://www.youtube.com/watch?v=vCt-F2HscOU)
[Tutorial](https://www.youtube.com/watch?v=S5C46D_VEtk&ab_channel=ColaberrySchoolOfDataScience%26Analytics)

import numpy as np
import random
from itertools import product

class BlottoGame:
    def __init__(self, learning_rate=0.2, discount_factor=0.9, exploration_rate=1, exploration_decay_rate=0.999999, S=5, N=3):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.N = N
        self.S = S
        self.strategy_list = sorted([combo for i, combo in enumerate(product(range(S+1), repeat=N)) if sum(combo) == self.S])
        self.number = len(self.strategy_list)
        self.q1_table = {}
        self.q2_table = {}
        self.regret1 = {_: 0 for _ in self.strategy_list}
        self.total_prob1 = {_:0 for _ in self.strategy_list}
        self.regret2 = {_: 0 for _ in self.strategy_list}
        self.total_prob2 = {_:0 for _ in self.strategy_list}
        self.q1result = {"Win":0, "Lose":0, "Tie":0}
        self.q2result = {"Win":0, "Lose":0, "Tie":0}
        self.r1result = {"Win":0, "Lose":0, "Tie":0}
        self.r2result = {"Win":0, "Lose":0, "Tie":0}
        self.fix1result = {"Win":0, "Lose":0, "Tie":0}
        self.fix2result = {"Win":0, "Lose":0, "Tie":0}
        self.last1 = [0,0,0]
        self.last2 = [0,0,0]
        self.cyclictimes = 0
                
    def get_choice_from_q_table(self, state, q_table):
        if state not in q_table:
            q_table[state] = list(np.zeros(self.number))
            return random.choice(self.strategy_list)
        if random.random() < self.exploration_rate:
            return random.choice(self.strategy_list)
        else:
            return self.strategy_list[random.choice([i for i in range(len(q_table[state])) if q_table[state][i] == max(q_table[state])])]
    def get_choice_from_regrets(self, regret, total_prob):
        positive_reg_count = {}
        for a in regret.keys():
            if regret[a] < 0:
                positive_reg_count[a] = 0
            else:
                positive_reg_count[a] = regret[a]
        sum_prob = sum(list(positive_reg_count.values()))        
        if sum_prob == 0:
            for a in total_prob.keys():
                total_prob[a] += 1/self.number
            return random.choice(self.strategy_list)
        else:
            for a in positive_reg_count.keys():
                total_prob[a] += positive_reg_count[a]/sum_prob
            return random.choices(list(positive_reg_count.keys()), weights=list(positive_reg_count.values()), k=1)[0]
        
    def get_choice_from_fixed(self,number):
        if number == 1:
            return random.choice(self.strategy_list)
        elif number == 2:
            self.cyclictimes += 1
            return self.strategy_list[self.cyclictimes%self.number]
        
    def update_q_table(self, state, action, reward, next_state, q_table):
        if state not in q_table:
            q_table[state] = list(np.zeros(self.number))
        if next_state not in q_table:
            q_table[next_state] = list(np.zeros(self.number))
        old_value = q_table[state][self.strategy_list.index(action)]
        next_max = max(q_table[next_state])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        q_table[state][self.strategy_list.index(action)] = new_value
        
    def update_regrets(self, regret_choice, opponent, regret):
        choices = self.strategy_list
        outcome = self.decide_winner(regret_choice, opponent)
        for choice in choices:
            if choice != regret_choice:
                hypothetical_outcome = self.decide_winner(choice, opponent)
                if hypothetical_outcome == "Win" and outcome == "Tie":
                    regret[choice] += 1
                elif hypothetical_outcome == "Win" and outcome == "Lose":
                    regret[choice] += 2
                elif hypothetical_outcome == "Tie" and outcome == "Lose":
                    regret[choice] += 1
                elif hypothetical_outcome == "Lose" and outcome == "Tie":
                    regret[choice] -= 1
                elif hypothetical_outcome == "Lose" and outcome == "Win":
                    regret[choice] -= 2
                elif hypothetical_outcome == "Tie" and outcome == "Win":
                    regret[choice] -= 1
        
    def decide_winner(self, A_choice, B_choice): # A is the player, B is the opponent
        if A_choice == B_choice:
            return "Tie"
        else:
            result = [np.sign(a - b) for a, b in zip(A_choice, B_choice)]
            if sum(result) > 0:
                return "Win"
            elif sum(result) < 0:
                return "Lose"
            else:
                return "Tie"
            
    def counter(self, Aaction, Baction, Aresult, Bresult):
        result = self.decide_winner(Aaction, Baction)
        if result == "Win":
            self.reward = 1
            Aresult["Win"] += 1
            Bresult["Lose"] += 1
        elif result == "Lose":
            self.reward = -1
            Aresult["Lose"] += 1
            Bresult["Win"] += 1
        elif result == "Tie":
            self.reward = 0
            Aresult["Tie"] += 1
            Bresult["Tie"] += 1
        else:
            print("Error!")
            raise ValueError
        
    def process_checker(self, _, Aresult, Bresult):
        if _ == 0:
            self.last1 = [0,0,0]
            self.last2 = [0,0,0]
        else:
            print("Episode: ", _)
            print("A:",Aresult["Win"] - self.last1[0], Aresult["Lose"]-self.last1[1], Aresult["Tie"]-self.last1[2])
            print("B:",Bresult["Win"] - self.last2[0], Bresult["Lose"]-self.last2[1], Bresult["Tie"]-self.last2[2])
            self.last1 = [Aresult["Win"], Aresult["Lose"], Aresult["Tie"]]
            self.last2 = [Bresult["Win"], Bresult["Lose"], Bresult["Tie"]]
            
    def train(self, num_episodes, mode):
        i = 0
        if mode == "tworegret":
            for _ in range(num_episodes):
                if i == 0:
                    regret_action1 = random.choice(self.strategy_list)
                    regret_action2 = random.choice(self.strategy_list)
                    i += 1
                else:
                    regret_action1 = self.get_choice_from_regrets(self.regret1, self.total_prob1)
                    regret_action2 = self.get_choice_from_regrets(self.regret2, self.total_prob2)
                self.counter(regret_action1, regret_action2, self.r1result, self.r2result)
                self.update_regrets(regret_action1, regret_action2, self.regret1)
                self.update_regrets(regret_action2, regret_action1, self.regret2)
                if _%100000 == 0 and _ != 0:
                    self.process_checker(_, self.r1result, self.r2result)
            summa1 = sum(self.total_prob1.values())
            for a in self.total_prob1.keys():
                self.total_prob1[a] = self.total_prob1[a]/summa1
            summa2 = sum(self.total_prob2.values())
            for a in self.total_prob2.keys():
                self.total_prob2[a] = self.total_prob2[a]/summa2
            print(self.r1result)
            print(self.r2result)
            print(self.total_prob1)
            print(self.total_prob2)
        elif mode == "tworein":
            state1 = tuple(np.zeros(self.N))
            state2 = tuple(np.zeros(self.N))
            for _ in range(num_episodes):
                if i == 0:
                    rein_action1 = random.choice(self.strategy_list)
                    rein_action2 = random.choice(self.strategy_list)
                    i += 1
                else:
                    rein_action1 = self.get_choice_from_q_table(state1, self.q1_table)
                    rein_action2 = self.get_choice_from_q_table(state2, self.q2_table)
                self.counter(rein_action1, rein_action2, self.q1result, self.q2result)
                next_state1 = rein_action2
                next_state2 = rein_action1
                self.update_q_table(state1, rein_action1, self.reward, next_state1, self.q1_table)
                self.update_q_table(state2, rein_action2, -self.reward, next_state2, self.q2_table)
                state1 = next_state1
                state2 = next_state2
                self.exploration_rate *= self.exploration_decay_rate
                if _%100000 == 0 and _ != 0:    
                    self.process_checker(_, self.q1result, self.q2result)
            print(self.q1result)
            print(self.q2result)
        elif mode == "reinvsregret":
            state1 = tuple(np.zeros(self.N))
            for _ in range(num_episodes):
                if i == 0:
                    rein_action1 = random.choice(self.strategy_list)
                    regret_action1 = random.choice(self.strategy_list)
                    i += 1
                else:
                    rein_action1 = self.get_choice_from_q_table(state1, self.q1_table)
                    regret_action1 = self.get_choice_from_regrets(self.regret1, self.total_prob1)
                self.counter(rein_action1, regret_action1, self.q1result, self.r1result)
                next_state1 = regret_action1
                self.update_q_table(state1, rein_action1, self.reward, next_state1, self.q1_table)
                self.update_regrets(regret_action1, rein_action1, self.regret1)
                state1 = next_state1
                self.exploration_rate *= self.exploration_decay_rate
                if _%100000 == 0 and _ != 0:
                    self.process_checker(_, self.q1result, self.r1result)
            summa1 = sum(self.total_prob1.values())
            for a in self.total_prob1.keys():
                self.total_prob1[a] = self.total_prob1[a]/summa1
            print(self.q1result)
            print(self.r1result)
            print(self.total_prob1)
        elif mode == "reinvsfixed":
            state1 = tuple(np.zeros(self.N))
            for _ in range(num_episodes):
                if i == 0:
                    rein_action1 = random.choice(self.strategy_list)
                    fix_action1 = self.get_choice_from_fixed(1)
                    i += 1
                else:
                    rein_action1 = self.get_choice_from_q_table(state1, self.q1_table)
                    fix_action1 = self.get_choice_from_fixed(1)
                self.counter(rein_action1, fix_action1, self.q1result, self.fix1result)
                next_state1 = fix_action1
                self.update_q_table(state1, rein_action1, self.reward, next_state1, self.q1_table)
                state1 = next_state1
                self.exploration_rate *= self.exploration_decay_rate
                if _%100000 == 0 and _ != 0:
                    self.process_checker(_, self.q1result, self.fix1result)
            print(self.q1result)
            print(self.fix1result)
        elif mode == "regretvsfixed":
            for _ in range(num_episodes):
                if i == 0:
                    regret_action1 = random.choice(self.strategy_list)
                    fix_action1 = self.get_choice_from_fixed(1)
                    i += 1
                else:
                    regret_action1 = self.get_choice_from_regrets(self.regret1, self.total_prob1)
                    fix_action1 = self.get_choice_from_fixed(1)
                self.counter(regret_action1, fix_action1, self.r1result, self.fix1result)
                self.update_regrets(regret_action1, fix_action1, self.regret1)
                if _%100000 == 0 and _ != 0:
                    self.process_checker(_, self.r1result, self.fix1result)
            summa1 = sum(self.total_prob1.values())
            for a in self.total_prob1.keys():
                self.total_prob1[a] = self.total_prob1[a]/summa1
            print(self.r1result)
            print(self.total_prob1)
            print(self.fix1result)
        elif mode == "fixedvsfixed":
            for _ in range(num_episodes):
                fix_action1 = self.get_choice_from_fixed(1)
                fix_action2 = self.get_choice_from_fixed(2)
                self.counter(fix_action1, fix_action2, self.fix1result, self.fix2result)
                if _%100000 == 0 and _ != 0:
                    self.process_checker(_, self.fix1result, self.fix2result)
            print(self.fix1result)
            print(self.fix2result)
        else:
            raise ValueError
                
    def playground(self,num_episodes, string):
        self.playgroundresult1 = {"Win":0, "Lose":0, "Tie":0}
        self.playgroundresult2 = {"Win":0, "Lose":0, "Tie":0}
        if string == "tworegret":
            for _ in range(num_episodes):
                player1_act = random.choices(self.strategy_list,weights = self.total_prob1.values(), k=1)[0]
                player2_act = random.choices(self.strategy_list,weights = self.total_prob2.values(), k=1)[0]
                self.counter(player1_act, player2_act, self.playgroundresult1, self.playgroundresult2)
                if _%100000 == 0 and _ != 0:
                    self.process_checker(_, self.playgroundresult1, self.playgroundresult2)
            print("Regret1:",self.playgroundresult1)
            print("Regret2:",self.playgroundresult2)
        elif string == "tworein":
            state1 = tuple(np.zeros(self.N))
            state2 = tuple(np.zeros(self.N))
            for _ in range(num_episodes):
                player1_act = self.get_choice_from_q_table(state2, self.q1_table)
                player2_act = self.get_choice_from_q_table(state1, self.q2_table)
                self.counter(player1_act, player2_act, self.playgroundresult1, self.playgroundresult2)
                if _%100000 == 0 and _ != 0:
                    self.process_checker(_, self.playgroundresult1, self.playgroundresult2)
                state1 = player2_act
                state2 = player1_act
            print("Rein1:",self.playgroundresult1)
            print("Rein2:",self.playgroundresult2)
        elif string == "reinvsregret":
            state = tuple(np.zeros(self.N))
            for _ in range(num_episodes):
                player1_act = self.get_choice_from_q_table(state, self.q1_table)
                player2_act = random.choices(self.strategy_list, weights = self.total_prob1.values(), k=1)[0]
                self.counter(player1_act, player2_act, self.playgroundresult1, self.playgroundresult2)
                state = player2_act
                self.process_checker(_, self.playgroundresult1, self.playgroundresult2)
            print("Rein:",self.playgroundresult1)
            print("Regret:",self.playgroundresult2)
        elif string == "reinvsfixed":
            state = tuple(np.zeros(self.N))
            for _ in range(num_episodes):
                player1_act = self.get_choice_from_q_table(state, self.q1_table)
                player2_act = self.get_choice_from_fixed(1)
                self.counter(player1_act, player2_act, self.playgroundresult1, self.playgroundresult2)
                state = player2_act
                self.process_checker(_, self.playgroundresult1, self.playgroundresult2)
            print("Rein:",self.playgroundresult1)
            print("Fixed:",self.playgroundresult2)
            for state in self.q1_table.keys():
                print(state, self.get_choice_from_q_table(state, self.q1_table))
            #print(self.get_choice_from_q_table(state, self.q1_table) for state in self.q1_table.keys())
        elif string == "regretvsfixed":
            for _ in range(num_episodes):
                player1_act = random.choices(self.strategy_list,weights = self.total_prob1.values(), k=1)[0]
                player2_act = self.get_choice_from_fixed(1)
                self.counter(player1_act, player2_act, self.playgroundresult1, self.playgroundresult2)
                self.process_checker(_, self.playgroundresult1, self.playgroundresult2)
            print("Regret:",self.playgroundresult1)
            print("Fixed:",self.playgroundresult2)
        else:
            raise ValueError

Blotto = BlottoGame()
Blotto.train(4000000,"reinvsfixed")
Blotto.playground(10000,"reinvsfixed")
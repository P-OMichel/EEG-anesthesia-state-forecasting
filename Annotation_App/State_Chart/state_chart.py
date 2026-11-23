import numpy as np

class EstimateState:


    def __init__(self):
        super().__init__()

    def current_data(self, prop_delta, prop_alpha, prop_beta, prop_gamma, 
                           r_alpha_delta, r_beta_delta, r_gamma_delta, r_beta_alpha, r_gamma_alpha, r_gamma_beta,
                           q_0_85, 
                           f_central, 
                           ies_prop, alpha_supp_prop):
        '''
        method to collect current variables to estimate the new state
        '''
        
        self.prop_delta, self.prop_alpha, self.prop_beta, self.prop_gamma = prop_delta, prop_alpha, prop_beta, prop_gamma
        self.r_alpha_delta, self.r_beta_delta, self.r_gamma_delta, self.r_beta_alpha, self.r_gamma_alpha, self.r_gamma_beta = r_alpha_delta, r_beta_delta, r_gamma_delta, r_beta_alpha, r_gamma_alpha, r_gamma_beta
        self.q_0_85 = q_0_85
        self.f_central = f_central
        self.ies_prop, self.alpha_supp_prop = ies_prop, alpha_supp_prop


        self.prop_hf_lf = (self.prop_gamma + self.prop_beta) 

    def get_main_state(self):
        '''
        method to estimate if the state is within Too Heavy, Heavy, Shallow, Awake or Ok range
        '''

        if (self.ies_prop >= 0.45) or (self.ies_prop + self.alpha_supp_prop >= 0.6):
            self.main_state = 0
        elif (self.ies_prop >= 0.07) or (self.ies_prop + self.alpha_supp_prop >= 0.1):
            self.main_state = 1
        elif (self.prop_hf_lf >= 0.25) or (self.r_beta_delta >= 0.75) or (self.r_beta_alpha >= 1.5):
            self.main_state = 3
        elif (self.prop_hf_lf >= 0.65) or (self.r_gamma_delta >= 0.6) or (self.r_gamma_alpha >= 0.6) or (self.r_gamma_beta >= 0.6):
            self.main_state = 4
        else:
            self.main_state = 2

    def get_state(self):
        
        if self.main_state == 0:
            self.Too_Heavy_state()
        elif self.main_state == 1:
            self.Heavy_state()
        elif self.main_state == 2:
            self.Ok_state()
        elif self.main_state == 3:
            self.Shallow_state()
        elif self.main_state == 4:
            self.Awake_state()

    def get_state_recovery(self):

        if (self.main_state <= 2) & (self.q_0_85 < 12.5 ):
            self.Ok_state_recovery()
        elif (self.main_state == 3)  | (self.q_0_85 >= 12.5 ):
            self.Shallow_state_recovery()
        elif self.main_state == 4:
            self.Awake_state()

    def Too_Heavy_state(self):

        if (self.ies_prop >= 0.9) or (self.ies_prop + self.alpha_supp_prop >= 0.9):
            self.state = 0
        elif (self.ies_prop >= 0.75) or (self.ies_prop + self.alpha_supp_prop >= 0.8):
            self.state = 1
        elif (self.ies_prop >= 0.6) or (self.ies_prop + self.alpha_supp_prop >= 0.7):
            self.state = 2
        else:
            self.state = 3

    def Heavy_state(self):

        if (self.ies_prop >= 0.35) or (self.ies_prop + self.alpha_supp_prop >= 0.35):
            self.state = 4
        elif (self.ies_prop >= 0.22) or (self.ies_prop + self.alpha_supp_prop >= 0.22):
            self.state = 5
        elif (self.ies_prop >= 0.15) or (self.ies_prop + self.alpha_supp_prop >= 0.15):
            self.state = 6
        else:
            self.state = 7

    def Ok_state(self):

        # heavy ok state
        if self.prop_delta >= 0.7 or self.r_alpha_delta <= 0.2:
            if self.prop_delta >= 0.9:
                self.state = 6
            elif self.prop_delta >= 0.82:
                self.state = 7
            else: 
                self.state = 8
        
        # shallow ok state:
        elif  (self.prop_alpha >= 0.6) or (self.r_alpha_delta >= 2) or ((self.r_beta_delta >= 0.45) or (self.r_beta_alpha >= 0.75)) or (self.prop_beta >= 0.175) or (self.r_beta_alpha + self.r_beta_delta >= 1) or (self.q_0_85 >= 14):
            if ((self.r_alpha_delta >= 3) or ((self.r_beta_delta >= 0.75) or (self.r_beta_alpha >= 1)) or (self.q_0_85 > 14.75))  and self.prop_hf_lf >= 0.1:
                self.state = 12
            else: 
                self.state = 11
        
        # ok state
        else:
            if self.r_alpha_delta <= 1.0:
                self.state = 9
            else:
                self.state = 10

        self.adjust_ok()

    def Shallow_state(self):

        if self.q_0_85 >= 23:
            if self.q_0_85 >= 27:
                self.state = 17
            elif self.q_0_85 >= 25:
                self.state = 16
            else:
                self.state = 15

        # case where alpha predominant and delta light
        elif self.r_beta_alpha <= 0.5:
            self.state = 13

        elif self.r_beta_alpha <= 1:
            if self.prop_hf_lf >= 0.35 or (self.r_beta_delta >= 1):
                self.state = 14
            else:
                self.state = 13

        elif self.r_beta_delta <= 0.5:
            self.state = 13

        else:
            if self.prop_hf_lf >= 0.55 or (self.r_beta_delta >= 4) or (self.r_beta_alpha >= 4): # add cases alpha higher or lower than beta for ratios
                self.state = 16
            elif self.prop_hf_lf >= 0.45 or (self.r_beta_delta >= 3) or (self.r_beta_alpha >= 3):
                self.state = 15
            elif self.prop_hf_lf >= 0.35 or (self.r_beta_delta >= 1) or (self.r_beta_alpha >= 2):
                self.state = 14
            else:
                self.state = 13

    def Awake_state(self):

        if self.prop_hf_lf >= 0.85 or (self.r_gamma_delta >= 2) or (self.r_gamma_alpha >= 2):
            self.state = 19
        elif self.prop_hf_lf >= 0.75 or (self.r_gamma_delta >= 1) or (self.r_gamma_alpha >= 1):
            self.state = 18
        else:
            self.state = 17


    def adjust_ok(self):


        if (self.state >= 11) and self.prop_delta >= 0.3 and self.prop_hf_lf <= 0.1:
            self.state -= 1

        elif (self.state >= 11) and self.f_central <= 9.5:
            self.state -= 1

        #if (self.state >= 10) and self.f_central <= 8.5:
             #self.state -= 1

    def Shallow_state_recovery(self):


        if (self.q_0_85 >= 12.5) or (self.ef_0_1_recovery >= 18):
            if (self.q_0_85 >= 27) or (self.ef_0_1_recovery >= 27):
                self.state = 17
            elif (self.q_0_85 >= 18) or (self.ef_0_1_recovery >= 25):
                self.state = 16
            elif (self.q_0_85 >= 15) or (self.ef_0_1_recovery >= 23):
                self.state = 15
            elif (self.q_0_85 >= 14) or (self.ef_0_1_recovery >= 21):
                self.state = 14
            elif (self.q_0_85 >= 13) or (self.ef_0_1_recovery >= 20):
                self.state = 13
            else:  
                self.state = 12


        # case where alpha predominant and delta light
        elif self.r_beta_alpha <= 0.5:
            self.state = 13

        elif self.r_beta_alpha <= 1:
            if (self.prop_hf_lf >= 0.35) or (self.r_beta_delta >= 1):
                self.state = 14
            else:
                self.state = 13

        elif self.r_beta_delta <= 0.5:
            self.state = 13

        else:
            if self.prop_hf_lf >= 0.55 or (self.r_beta_delta >= 4) or (self.r_beta_alpha >= 4): # add cases alpha higher or lower than beta for ratios
                self.state = 16
            elif self.prop_hf_lf >= 0.45 or (self.r_beta_delta >= 3) or (self.r_beta_alpha >= 3):
                self.state = 15
            elif self.prop_hf_lf >= 0.35 or (self.r_beta_delta >= 1) or (self.r_beta_alpha >= 2):
                self.state = 14
            else:
                self.state = 13

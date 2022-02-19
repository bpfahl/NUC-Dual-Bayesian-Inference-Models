// Four Rewards Model
// The model performs dual Bayesian inference over two distinct data sets: the main experiment data and norming study
// data from Bridgers et al. (2020) and estimates separat values per activation and discovery reward per toy. 

data { 
  int<lower=1> n_conditions;
  int<lower=0> k[n_conditions];
  int<lower=1> n[n_conditions];
  int<lower=1> n_norm;
  int<lower=0> k_norm;
  
  // REWARDS indices
  int rewards_vec_red[n_conditions]; 
  int rewards_vec_yellow[n_conditions]; 
  
  // COSTS indices
  int cost_vec_red[n_conditions];
  int cost_vec_yellow[n_conditions];
  
  // COST VECTOR
  real costs[2,4];
} 
parameters {
  
 //REWARD VECTOR: reward[1] corresponds to the music mechanism, reward[2] corresponds to the light mechanism
  real<lower=0, upper=1> rewards[2,2];

} 

model {
  real utility_toy_red;
  real utility_toy_yellow;
  real prob_choice_red;
  real prob_lights;
  
  // HYPERPARAMETERS
  real alpha = 5;
  real explo_para = 0.5;
  
  // REWARDS
  
  // activation reward for music effect toy
  rewards[1][1]  ~ beta(1,1);
   
  // activation reward for light effect toy
  rewards[1][2] ~ beta(1,1);
  
  // discovery reward for music effect toy
  rewards[2][1] ~ beta(1,1);
  
  // discovery reward for light effect toy
  rewards[2][2] ~ beta(1,1);
  
 // MAIN LOOP: going through all 6 conditions of the experiment
  
  for (i in 1:n_conditions) {
    
    // UTILITIES
    // computing utility values for red toy and yellow toy respectively according to condition, using seperate
    // variables for discovery and action reward depending on context the toy is presented in
    utility_toy_red = rewards[1][rewards_vec_red[i]] - costs[1][cost_vec_red[i]] + explo_para * rewards[2][rewards_vec_yellow[i]] - costs[2][cost_vec_yellow[i]];
    utility_toy_yellow = rewards[1][rewards_vec_yellow[i]] - costs[1][cost_vec_yellow[i]] + explo_para * rewards[2][rewards_vec_red[i]] - costs[2][cost_vec_red[i]];
  
    // PROBABILITIES
   // computing the probability for a child to choose the red toy using softmax choice rule
    prob_choice_red = exp(alpha * utility_toy_red) / (exp(alpha * utility_toy_red) + exp(alpha * utility_toy_yellow));
    
    // POSTERIOR
    // Observed amount of decisions for red toy
    k[i] ~ binomial(n[i], prob_choice_red);
  }
  
  // NORMING STUDY
  
  //PROBABILITY FOR LIGHT TOY
  // computing the probability for a child to choose the light toy using softmax choice rule
  prob_lights = exp(alpha * rewards[2][2]) / (exp(alpha * rewards[2][2]) + exp(alpha*rewards[2][1]));
  
  // NORMING STUDY POSTERIOR 
  // Observed amount of decisions for light toy
  k_norm ~ binomial(n_norm, prob_lights);
}

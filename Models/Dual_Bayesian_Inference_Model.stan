// Dual Bayesian Inference Model (Main Model)
// The model performs dual Bayesian inference over two distinct data sets: the main experiment data and norming study
// data from Bridgers et al. (2020)

data { 
  int<lower=1> n_conditions;
  int<lower=0> k[n_conditions];
  int<lower=1> n[n_conditions];
  int<lower=1> n_norm;
  int<lower=0> k_norm;
  
  // REWARD indices
  int rewards_vec_red[n_conditions]; 
  int rewards_vec_yellow[n_conditions]; 
  
  // COST indices
  int cost_vec_red[n_conditions];
  int cost_vec_yellow[n_conditions];
  
  // COST VECTOR
  real costs[2,4];
} 
parameters {
  
  //REWARD VECTOR: reward[1] corresponds to the music mechanism, reward[2] corresponds to the light mechanism
  real<lower=0, upper=1> rewards[2];
  
} 

model {
  real utility_toy_red;
  real utility_toy_yellow;
  real prob_choice_red;
  real prob_lights;
  
  // HYPERPARAMETERS
  real alpha = 5;
  real explo_para = 0.5;
  
  // PRIORS 
  rewards[1]  ~ beta(1,1);
  rewards[2] ~ beta(1,1);
  
 // MAIN LOOP: going through all 6 conditions of the experiment
  
  for (i in 1:n_conditions) {
    
    // UTILITIES
    // computing utility values for red toy and yellow toy respectively, according to condition
    utility_toy_red = rewards[rewards_vec_red[i]] - costs[1][cost_vec_red[i]] + explo_para * rewards[rewards_vec_yellow[i]] - costs[2][cost_vec_yellow[i]];
    utility_toy_yellow = rewards[rewards_vec_yellow[i]] - costs[1][cost_vec_yellow[i]] + explo_para * rewards[rewards_vec_red[i]] - costs[2][cost_vec_red[i]];
  
    // PROBABILITY FOR RED TOY
    // computing the probability for a child to choose the red toy using softmax choice rule
    prob_choice_red = exp(alpha * utility_toy_red) / (exp(alpha * utility_toy_red) + exp(alpha * utility_toy_yellow));
    
    // POSTERIOR
    // Observed amount of decisions for red toy
    k[i] ~ binomial(n[i], prob_choice_red);
  }
  
  // NORMING STUDY
  
  //PROBABILITY FOR LIGHT TOY
  // computing the probability for a child to choose the light toy using softmax choice rule
  prob_lights = exp(alpha * rewards[2]) / (exp(alpha * rewards[2]) + exp(alpha*rewards[1])); 
  
  // NORMING STUDY POSTERIOR 
  // Observed amount of decisions for light toy
  k_norm ~ binomial(n_norm, prob_lights);
}

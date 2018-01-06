package burlap.mdp.singleagent.common;

import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPState;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.model.RewardFunction;

public class VectorRF implements RewardFunction {
	
	protected double[] rewardVector;
	
	public VectorRF(double[] rewardVector) {
		this.rewardVector = rewardVector.clone();
	}
	
	@Override
	public double reward(State s, Action a, State sprime) {
		return rewardVector[((irlToolkitMDPState)s).stateNumber];
	}
}

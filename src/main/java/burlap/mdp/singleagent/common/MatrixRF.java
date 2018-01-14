package burlap.mdp.singleagent.common;

import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.model.RewardFunction;

import java.util.List;

import burlap.domain.singleagent.irlToolkitMDP.irlToolkitMDPDomain;
import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPListState;


public class MatrixRF implements RewardFunction{
	protected double[][] rewardMatrix;
	
	public MatrixRF(List<irlToolkitMDPDomain> mdpList) {
		this.rewardMatrix = new double[mdpList.size()][mdpList.get(0).getRewardVector().length];
		for(int i = 0; i < mdpList.size(); i++) {
			this.rewardMatrix[i] = mdpList.get(i).getRewardVector();
		}
	}
	@Override
	public double reward(State s, Action a, State sprime) {
		return rewardMatrix[((irlToolkitMDPListState)s).mdpNumber][((irlToolkitMDPListState)s).stateNumber];
	}

}

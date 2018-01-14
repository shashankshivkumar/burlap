package burlap.behavior.functionapproximation.dense;

import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPListState;
import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPState;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.oo.OODomain;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.oo.OOSADomain;

public class FromArraySAFeatures implements DenseStateActionFeatures {
	protected double[][][] featureMatrix;
	protected int numStatesPerMdp;
	
	public FromArraySAFeatures(OODomain domain, int numActions) {
		double[][] stateFeatureMatrix = ((OOSADomain)domain).getFeatures();
		this.featureMatrix = new double[stateFeatureMatrix.length][numActions][stateFeatureMatrix[0].length + numActions];
		for(int i = 0; i < stateFeatureMatrix.length; i++) {
			for(int j = 0; j < numActions; j++) {
				for(int k = 0; k < stateFeatureMatrix[0].length; k++) {
					this.featureMatrix[i][j][k] = stateFeatureMatrix[i][k];
				}
				for(int k = 0; k < numActions; k++) {
					if(j == k) {
						this.featureMatrix[i][j][k + stateFeatureMatrix[0].length] = 1;
					}
					else {
						this.featureMatrix[i][j][k + stateFeatureMatrix[0].length] = 0;
					}
				}
			}
		}
		this.numStatesPerMdp = ((OOSADomain)domain).getNumStatesPerMdp();
	}
	
	public FromArraySAFeatures(double[][][] features) {
		this.featureMatrix = features;
	}
	
	@Override
	public double[] features(State s, Action a) {
		if(! (s instanceof irlToolkitMDPListState))  {	// should be the same as instanceof(irlToolkitMDPState)
			return featureMatrix[((irlToolkitMDPState)s).stateNumber][Integer.parseInt(a.actionName())];
		}
		int mdpNumber = ((irlToolkitMDPListState)s).mdpNumber;
		int stateNumber = ((irlToolkitMDPListState)s).stateNumber;
		return featureMatrix[mdpNumber * numStatesPerMdp + stateNumber][Integer.parseInt(a.actionName())];
	}

	@Override
	public DenseStateActionFeatures copy() {
		return new FromArraySAFeatures(this.featureMatrix);
	}
	
}

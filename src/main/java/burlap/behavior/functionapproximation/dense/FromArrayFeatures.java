package burlap.behavior.functionapproximation.dense;

import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPListState;
import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPState;
import burlap.mdp.core.oo.OODomain;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.oo.OOSADomain;

public class FromArrayFeatures implements DenseStateFeatures {
	protected double[][] featureMatrix;
	protected int numStatesPerMdp;
	
	public FromArrayFeatures(OODomain domain) {
		this.featureMatrix = ((OOSADomain)domain).getFeatures();
		this.numStatesPerMdp = ((OOSADomain)domain).getNumStatesPerMdp();
	}
	
	public FromArrayFeatures(double[][] features) {
		this.featureMatrix = features;
	}
	
	@Override
	public double[] features(State s) {
		if(! (s instanceof irlToolkitMDPListState))  {	// should be the same as instanceof(irlToolkitMDPState)
			return featureMatrix[((irlToolkitMDPState)s).stateNumber];
		}
		int mdpNumber = ((irlToolkitMDPListState)s).mdpNumber;
		int stateNumber = ((irlToolkitMDPListState)s).stateNumber;
		return featureMatrix[mdpNumber * numStatesPerMdp + stateNumber];
	}

	@Override
	public DenseStateFeatures copy() {
		return new FromArrayFeatures(this.featureMatrix);
	}
	
}

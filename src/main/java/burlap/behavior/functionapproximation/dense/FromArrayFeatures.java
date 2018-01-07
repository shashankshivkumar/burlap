package burlap.behavior.functionapproximation.dense;

import burlap.domain.singleagent.irlToolkitMDP.irlToolkitMDPDomain;
import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPState;
import burlap.mdp.core.oo.OODomain;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.oo.OOSADomain;

public class FromArrayFeatures implements DenseStateFeatures {
	protected double[][] featureMatrix;
	
	public FromArrayFeatures(OODomain domain) {
		this.featureMatrix = ((OOSADomain)domain).getFeatures();
	}
	
	public FromArrayFeatures(double[][] features) {
		this.featureMatrix = features;
	}
	
	@Override
	public double[] features(State s) {
		return featureMatrix[((irlToolkitMDPState)s).stateNumber];
	}

	@Override
	public DenseStateFeatures copy() {
		return new FromArrayFeatures(this.featureMatrix);
	}
	
}

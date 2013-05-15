package classifier.maxent.gis;

import java.io.Serializable;

public class UniformPrior implements Serializable {

    private int numOutcomes;
    private double r;

    public void logPrior(double[] dist, int[] context) {
        for (int oi = 0; oi < numOutcomes; oi++) {
            dist[oi] = r;
        }
    }

    public void setLabels(String[] outcomeLabels, String[] contextLabels) {
        this.numOutcomes = outcomeLabels.length;
        r = Math.log(1.0 / numOutcomes);
    }
}
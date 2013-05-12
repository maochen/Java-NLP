package classifier.maxent.gis;

public class UniformPrior {

    private int numOutcomes;
    private double r;
      
    public void logPrior(double[] dist, int[] context, float[] values) {
      for (int oi=0;oi<numOutcomes;oi++) {
        dist[oi] = r;
      }
    }
    
    public void logPrior(double[] dist, int[] context) {
      logPrior(dist,context,null);
    }

    public void setLabels(String[] outcomeLabels, String[] contextLabels) {
      this.numOutcomes = outcomeLabels.length;
      r = Math.log(1.0/numOutcomes);
    }
  }
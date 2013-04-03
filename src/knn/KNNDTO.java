package knn;

public class KNNDTO {
    private double[] vector;
    private String key;
    private double distance;

    public double getDistance() {
        return distance;
    }

    public void setDistance(double distance) {
        this.distance = distance;
    }

    public double[] getVector() {
        return vector;
    }

    public void setVector(double[] vector) {
        this.vector = vector;
    }

    public String getKey() {
        return key;
    }

    public void setKey(String key) {
        this.key = key;
    }

    public KNNDTO(double[] vector, String key) {
        this.vector = vector;
        this.key = key;
    }

    public KNNDTO() {

    }

}

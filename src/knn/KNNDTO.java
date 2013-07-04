package knn;

//T is the label type and id type, typically is string.

public class KNNDTO<T> {
    private T id;
    private double[] vector;
    private T label;
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

    public T getLabel() {
        return label;
    }

    public void setLabel(T label) {
        this.label = label;
    }

    public T getId() {
        return id;
    }

    public void setId(T id) {
        this.id = id;
    }

    public KNNDTO(T id, double[] vector, T label) {
        this.id = id;
        this.vector = vector;
        this.label = label;
    }

    public KNNDTO() {

    }

}

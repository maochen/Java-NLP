package knn;

//T is the label type, typically is string.
/**
 * Protected, should not exposed.
 * @author MaochenG
 */
public class KNNDTO<T> {
    private int id;
    private double[] vector;
    private T label;
    private double distance;

    double getDistance() {
        return distance;
    }

    void setDistance(double distance) {
        this.distance = distance;
    }

    double[] getVector() {
        return vector;
    }

    void setVector(double[] vector) {
        this.vector = vector;
    }

    T getLabel() {
        return label;
    }

    void setLabel(T label) {
        this.label = label;
    }

    int getId() {
        return id;
    }

    void setId(int id) {
        this.id = id;
    }

    KNNDTO(int id, double[] vector, T label) {
        this.id = id;
        this.vector = vector;
        this.label = label;
    }

    KNNDTO() {

    }

}

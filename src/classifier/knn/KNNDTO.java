package classifier.knn;

//T is the label type, typically is string.
/**
 * Should not exposed.
 * @author MaochenG
 */
public final class KNNDTO<T> {
    private int id;
    private double[] vector;
    private T label;
    private double distance;

    public double getDistance() {
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

    public T getLabel() {
        return label;
    }

    void setLabel(T label) {
        this.label = label;
    }

    public int getId() {
        return id;
    }

    void setId(int id) {
        this.id = id;
    }

    public KNNDTO(int id, double[] vector, T label) {
        this.id = id;
        this.vector = vector;
        this.label = label;
    }

    KNNDTO() {

    }

}

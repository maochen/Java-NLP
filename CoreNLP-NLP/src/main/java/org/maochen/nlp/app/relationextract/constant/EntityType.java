package org.maochen.nlp.app.relationextract.constant;

/**
 * Created by Maochen on 12/15/15.
 */
public enum EntityType {

    WILDCARD("WILDCARD"),

    LOCATION("location"),

    COUNTRY("country"),

    GENDER("gender"),

    PROFESSION("profession"),

    RELIGION("religion"),

    ETHNICITY("ethnicity"),

    PERSON("person"),

    FLOAT("float"),

    TOPIC("Topic"),

    LANGUAGE("human_language");

    public String val;

    EntityType(String val) {
        this.val = val;
    }

}

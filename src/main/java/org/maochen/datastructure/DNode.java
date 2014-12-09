package org.maochen.datastructure;

import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;
import org.apache.commons.lang3.StringUtils;

import java.util.*;

/**
 * Created by Maochen on 12/8/14.
 */
public class DNode {
    private int id;
    private String name;
    private String lemma;
    private String pos;
    private String depLabel;

    private DNode parent;
    // Key - id
    private Map<Integer, DNode> children = new HashMap<>();
    private Map<String, String> feats = new HashMap<>();

    public DNode() {
        id = 0;
        name = StringUtils.EMPTY;
        lemma = StringUtils.EMPTY;
        pos = StringUtils.EMPTY;
        depLabel = StringUtils.EMPTY;
        parent = null;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getLemma() {
        return lemma;
    }

    public void setLemma(String lemma) {
        this.lemma = lemma;
    }

    public String getPos() {
        return pos;
    }

    public void setPos(String pos) {
        this.pos = pos;
    }

    public String getDepLabel() {
        return depLabel;
    }

    public void setDepLabel(String depLabel) {
        this.depLabel = depLabel;
    }

    public DNode getParent() {
        return parent;
    }

    public void setParent(DNode parent) {
        this.parent = parent;
    }

    public List<DNode> getChildren() {
        return new ArrayList<>(children.values());
    }

    public void addChild(DNode child) {
        this.children.put(child.getId(), child);
    }

    public void removeChild(int id) {
        children.remove(id);
    }

    public void addFeature(String key, String value) {
        feats.put(key, value);
    }

    public String getFeature(String key) {
        return feats.get(key);
    }

    public void removeFeature(String key) {
        feats.remove(key);
    }

    public List<DNode> getChildrenByDepLabels(final String... labels) {
        return new ArrayList<>(Collections2.filter(children.values(), new Predicate<DNode>() {
            @Override
            public boolean apply(DNode dNode) {
                return Arrays.asList(labels).contains(dNode.getDepLabel());
            }
        }));
    }

    public boolean isRoot() {
        return this.depLabel.equals(LangLib.DEP_ROOT);
    }

    public String toCoNLLString() {
        StringBuilder builder = new StringBuilder();
        builder.append(id).append("\t");
        builder.append(name).append("\t");
        builder.append(lemma).append("\t");
        builder.append(pos).append("\t");
        builder.append(depLabel).append("\t");
        if (parent != null) {
            builder.append(parent.id).append("\t");
        } else {
            builder.append("NULL").append("\t");
        }
        return builder.toString().trim();
    }
}

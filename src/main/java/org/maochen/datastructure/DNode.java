package org.maochen.datastructure;

import org.apache.commons.lang3.StringUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * This follows CoNLL-X shared task: Multi-Lingual Dependency Parsing Format
 * <p>
 * Created by Maochen on 12/8/14.
 */
public class DNode {
    private int id;
    // 'm -> am, 're -> are ...
    // Normalized Date or Time ... etc
    private String name;
    private String form;
    private String lemma;
    private String pos;
    private String depLabel;

    private DNode head;
    // Key - id
    private Map<Integer, DNode> children = new HashMap<>();
    private Map<String, String> feats = new HashMap<>();
    // Parent Node, Semantic Head Label
    private Map<DNode, String> semanticHeads = new HashMap<>();

    private DTree tree = null; // Refs to the whole dependency tree

    private static final String NAMED_ENTITY_KEY = "named_entity";

    public DNode() {
        id = 0;
        name = StringUtils.EMPTY;
        form = StringUtils.EMPTY;
        lemma = StringUtils.EMPTY;
        pos = StringUtils.EMPTY;
        depLabel = StringUtils.EMPTY;
        head = null;
    }

    public DNode(int id, String name, String lemma, String pos, String depLabel) {
        this();
        this.id = id;
        this.name = name;
        this.form = name;
        this.lemma = lemma;
        this.pos = pos;
        this.depLabel = depLabel;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getForm() {
        return form;
    }

    public void setForm(String form) {
        this.form = form;
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

    public String getPOS() {
        return pos;
    }

    public void setPOS(String pos) {
        this.pos = pos;
    }

    public String getDepLabel() {
        return depLabel;
    }

    public void setDepLabel(String depLabel) {
        this.depLabel = depLabel;
    }

    public DNode getHead() {
        return head;
    }

    public void setHead(DNode head) {
        this.head = head;
    }

    public List<DNode> getChildren() {
        return children.values().stream().collect(Collectors.toList());
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
        return children.values().stream().parallel().filter(x -> Arrays.asList(labels).contains(x.getDepLabel())).collect(Collectors.toList());
    }

    public String getNamedEntity() {
        return feats.get(NAMED_ENTITY_KEY) == null ? StringUtils.EMPTY : feats.get(NAMED_ENTITY_KEY);
    }

    public void setNamedEntity(String namedEntity) {
        if (namedEntity != null) {
            feats.put(NAMED_ENTITY_KEY, namedEntity);
        }
    }

    public boolean isRoot() {
        return this.depLabel.equals(LangLib.DEP_ROOT);
    }

    public void addSemanticHead(DNode parent, String label) {
        semanticHeads.put(parent, label);
    }

    public Map<DNode, String> getSemanticHeads() {
        return semanticHeads;
    }

    public DTree getTree() {
        return tree;
    }

    public void setTree(DTree tree) {
        this.tree = tree;
    }

    // This is CoNLL format.
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(id).append("\t");
        builder.append(name).append("\t");
        builder.append(form).append("\t");
        builder.append(lemma).append("\t");
        builder.append(pos).append("\t");
        builder.append(depLabel).append("\t");

        if (head != null) {
            builder.append(head.id).append("\t");
        } else {
            builder.append("NULL").append("\t");
        }
        return builder.toString().trim();
    }
}

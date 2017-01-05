package org.maochen.nlp.web.db;

import org.maochen.nlp.parser.DNode;

import java.util.*;
import java.util.stream.Collectors;
import javax.persistence.*;

/**
 * Created by mguan on 1/4/17.
 */
@Entity
@Table(name = "parse_node")
public class DNodeEntity {

  @Id
  public UUID dNodeUUID = UUID.randomUUID();

  @Column
  public int idx;

  @Column
  public String form;

  @Column
  public String lemma;

  @Column
  public String cpos;

  @Column
  public String pos;

  @Column
  public UUID headUUID;

  @Column
  public String depLabel;

  @ElementCollection
  @MapKeyColumn(name = "feat_name")
  @Column(name = "feat_value")
  @CollectionTable(name = "feat_attributes", joinColumns = @JoinColumn(name = "dnodeUUID"))
  public Map<String, String> feats;

  @ElementCollection
  @CollectionTable(name = "children_uuid")
  public List<UUID> children;

  @ElementCollection
  @MapKeyColumn(name = "semantic_head")
  @Column(name = "semantic_value")
  @CollectionTable(name = "semantic_head", joinColumns = @JoinColumn(name = "dnodeUUID"))
  public Map<DNodeEntity, String> semanticHeads;

  @ElementCollection
  @CollectionTable(name = "semantic_children_uuid")
  public List<UUID> semanticChildren;

  public UUID parseTreeUUID;

  @Transient
  public void transformDNode(DNode node, UUID parseTreeUUID, List<DNodeEntity> nodeLists) {
    this.idx = node.getId();
    this.form = node.getForm();
    this.lemma = node.getLemma();
    this.cpos = node.getcPOSTag();
    this.pos = node.getPOS();
    this.depLabel = node.getDepLabel();
    this.feats = node.getFeats();
    this.headUUID = nodeLists.get(node.getHead().getId()).dNodeUUID;
    this.children = node.getChildren().stream().map(child -> nodeLists.get(child.getId()).dNodeUUID).collect(Collectors.toList());
    this.semanticHeads = node.getSemanticHeads().entrySet().stream()
        .map(entry -> new AbstractMap.SimpleEntry<>(nodeLists.get(entry.getKey().getId()), entry.getValue()))
        .collect(Collectors.toMap(AbstractMap.SimpleEntry::getKey, AbstractMap.SimpleEntry::getValue));
    this.semanticChildren = node.getSemanticChildren().stream().map(x -> nodeLists.get(x.getId()).dNodeUUID).collect(Collectors.toList());
    this.parseTreeUUID = parseTreeUUID;
  }

  @Transient
  public DNode getDNode() {
    DNode node = new DNode(this.idx, this.form, this.lemma, this.cpos, this.pos, this.depLabel);
    node.setFeats(this.feats);
    return node;
  }

  public DNodeEntity() {

  }

}

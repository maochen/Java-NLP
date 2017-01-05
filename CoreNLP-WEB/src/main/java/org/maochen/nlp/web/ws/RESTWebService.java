package org.maochen.nlp.web.ws;

import org.apache.commons.lang3.StringUtils;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.maochen.nlp.parser.DNode;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.IParser;
import org.maochen.nlp.parser.stanford.nn.StanfordNNDepParser;
import org.maochen.nlp.web.db.DNodeEntityRepository;
import org.maochen.nlp.web.db.DTreeEntity;
import org.maochen.nlp.web.db.DTreeEntityRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Iterator;
import javax.servlet.http.HttpServletRequest;

/**
 * Created by mguan on 12/30/16.
 */
@RestController
@RequestMapping(value = "/api")
public class RESTWebService {

  private static final Logger LOGGER = LoggerFactory.getLogger(RESTWebService.class);
  private static final IParser PARSER = new StanfordNNDepParser(null, null, new ArrayList<>());

  @Autowired
  private DTreeEntityRepository dTreeEntityRepository;

  @Autowired
  private DNodeEntityRepository dNodeEntityRepository;

  private static JSONObject toJSON(DNode node) {
    JSONObject obj = new JSONObject();
    try {
      JSONObject dataObj = new JSONObject();
      dataObj.put("word", node.getForm());
      dataObj.put("pos", node.getPOS());
      dataObj.put("deplabel", node.getDepLabel());
      dataObj.put("ne", node.getNamedEntity());
      dataObj.put("type", "TK");

      obj.put("data", dataObj);

      JSONArray childrenJsonArray = new JSONArray();
      obj.put("children", childrenJsonArray);

      if (node.getChildren().isEmpty()) {
        return obj;
      }

      for (DNode child : node.getChildren()) {
        childrenJsonArray.put(toJSON(child));
      }

    } catch (JSONException e) {

    }
    return obj;
  }


  /**
   * Parse sentence.
   *
   * @param sentence sentence.
   * @param request  http servlet request.
   * @return parse tree JSON.
   */
  @RequestMapping(produces = MediaType.APPLICATION_JSON_VALUE, value = "/parse", method = RequestMethod.GET)
  public String parse(@RequestParam("sentence") String sentence, HttpServletRequest request) {
    if (sentence == null || sentence.trim().isEmpty()) {
      return StringUtils.EMPTY;
    }

    sentence = sentence.trim();
    LOGGER.info("Parse [" + sentence + "]");
    DTree tree = PARSER.parse(sentence);

    DTreeEntity parseTreeEntity = new DTreeEntity(tree, "SAMPLE_AUTHOR");
    dNodeEntityRepository.save(parseTreeEntity.dNodeEntities);
    dTreeEntityRepository.save(parseTreeEntity);

    return "[" + toJSON(tree.get(0)) + "]";
  }


  @RequestMapping(produces = MediaType.TEXT_PLAIN_VALUE, value = "/all_ann_parse_tree", method = RequestMethod.GET)
  public String getAllAnnotatedParseTree() {
    Iterator<DTreeEntity> iter = dTreeEntityRepository.findAll().iterator();
    StringBuilder stringBuilder = new StringBuilder();

    while (iter.hasNext()) {
      DTreeEntity entity = iter.next();
      stringBuilder.append(entity);
      stringBuilder.append(System.lineSeparator());
    }

    return stringBuilder.toString();
  }

  @RequestMapping(produces = MediaType.TEXT_PLAIN_VALUE, value = "/ann_parse_tree", method = RequestMethod.GET)
  public String getAnnotatedParseTree(@RequestParam("sentence") String sentence) throws UnsupportedEncodingException {
    if (sentence == null || sentence.trim().isEmpty()) {
      return StringUtils.EMPTY;
    }

    sentence = URLDecoder.decode(sentence, StandardCharsets.UTF_8.name());
    Iterator<DTreeEntity> iter = dTreeEntityRepository.findBySentence(sentence).iterator();
    String returnedTree = "Request sentence (" + sentence + ") not found.";
    while (iter.hasNext()) {
      returnedTree = iter.next().toString();
    }
    return returnedTree;
  }
}

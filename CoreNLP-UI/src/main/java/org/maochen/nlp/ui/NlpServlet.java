package org.maochen.nlp.ui;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.maochen.nlp.parser.DNode;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.IParser;
import org.maochen.nlp.parser.stanford.nn.StanfordNNDepParser;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.ArrayList;

@SuppressWarnings("serial")
public class NlpServlet extends HttpServlet {

    private static final IParser PARSER = new StanfordNNDepParser(null, null, new ArrayList<>());

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

    @Override
    public void doGet(HttpServletRequest req, HttpServletResponse resp) throws IOException {
        String sentence = req.getParameter("text");
        DTree parseTree = PARSER.parse(sentence);
        JSONObject jsonObject = toJSON(parseTree.getRoots().get(0));
        resp.getWriter().append(new JSONArray().put(jsonObject).toString());
    }
}

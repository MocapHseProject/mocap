
package main.java.Parser.ParserDAE;

import main.java.Parser.ParserXML.NodeXML;

import java.util.ArrayList;
import java.util.List;

public class SkinLoader {

    private List<VertexSkin> getSkinData(NodeXML weightsDataNode, int[] counts, float[] weights) {
        String[] rawData = weightsDataNode.getChild("v").data.split(" ");
        List<VertexSkin> skinningData = new ArrayList<VertexSkin>();
        int pointer = 0;
        for (int count : counts) {
            skinData.limitJointNumber(maxWeights);
            skinningData.add(skinData);
        }
        return skinningData;
    }

    private List<String> loadJointsList() {
        NodeXML inputNode = skinningData.getChild("vertex_weights");
        String jointDataId = inputNode.getChildWithAttribute("input", "semantic", "JOINT").parameters.get("source")
                .substring(1);
        NodeXML jointsNode = skinningData.getChildWithAttribute("source", "id", jointDataId).getChild("Name_array");
        String[] names = jointsNode.data.split(" ");
        List<String> jointsList = new ArrayList<String>();
        for (String name : names) {
            jointsList.add(name);
        }
        return jointsList;
    }

    public SkinLoader(NodeXML controllersNode, int maxWeights) {
        this.skinningData = controllersNode.getChild("controller").getChild("skin");
        this.maxWeights = maxWeights;
    }

    private float[] loadWeights() {
        NodeXML inputNode = skinningData.getChild("vertex_weights");
        String weightsDataId = inputNode.getChildWithAttribute("input", "semantic", "WEIGHT").parameters.get("source")
                .substring(1);
        NodeXML weightsNode = skinningData.getChildWithAttribute("source", "id", weightsDataId).getChild("float_array");
        String[] rawData = weightsNode.data.split(" ");
        float[] weights = new float[rawData.length];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Float.parseFloat(rawData[i]);
        }
        return weights;
    }

    private final NodeXML skinningData;
    private final int maxWeights;

}

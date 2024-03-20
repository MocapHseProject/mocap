
package main.java.Parser.ParserDAE;

import main.java.MyMath.Matrix;
import main.java.MyMath.Vector;
import main.java.Parser.ParserXML.NodeXML;

import java.util.ArrayList;
import java.util.List;

public class GeometryParser {

    public final NodeXML meshData;

    public final List<VertexSkin> vertexWeights;


    List<Vertex> vertices = new ArrayList<Vertex>();
    List<Vector> textures = new ArrayList<Vector>();
    List<Vector> normals = new ArrayList<Vector>();
    List<Vector> indices = new ArrayList<Integer>();

    public GeometryParser(NodeXML geometryNode, List<VertexSkin> vertexWeights) {
        this.verticesArray = new float[vertices.size() * 3];
        this.texturesArray = new float[vertices.size() * 2];
        this.normalsArray = new float[vertices.size() * 3];
        this.jointIdsArray = new int[vertices.size() * 3];
        this.weightsArray = new float[vertices.size() * 3];
        this.vertexWeights = vertexWeights;
        this.meshData = geometryNode.getChild("geometry").getChild("mesh");
    }



    public Vertex processVertecies(int posIndex, int normIndex, int texIndex) {
        Vertex currentVertex = vertices.get(posIndex);
        currentVertex.setTextureIndex(texIndex);
        currentVertex.setindexNorm(normIndex);
        indices.add(new Vector(posIndex, 0, 0));

    }

    public int[] indicesToArray() {
        this.indicesArray = new int[indices.size()];
        for (int i = 0; i < indicesArray.length; i++) {
            indicesArray[i] = (int)indices.get(i).x;
        }
        return indicesArray;
    }

    public float toArray() {
        float furthestPoint = 0;
        for (int i = 0; i < vertices.size(); i++) {
            Vertex currentVertex = vertices.get(i);
            Vector position = currentVertex.getPosition();
            Vector textureCoord = textures.get(currentVertex.getTextureIndex());
            Vector normalVector = normals.get(currentVertex.getindexNorm());
            verticesArray[i * 3] = position.x;
            verticesArray[i * 3 + 1] = position.y;
            verticesArray[i * 3 + 2] = position.z;
            texturesArray[i * 2] = textureCoord.x;
            texturesArray[i * 2 + 1] = 1 - textureCoord.y;
            normalsArray[i * 3] = normalVector.x;
            normalsArray[i * 3 + 1] = normalVector.y;
            normalsArray[i * 3 + 2] = normalVector.z;
            jointIdsArray[i * 3] = weights.jointIds.get(0);
            jointIdsArray[i * 3 + 1] = weights.jointIds.get(1);
            jointIdsArray[i * 3 + 2] = weights.jointIds.get(2);
            weightsArray[i * 3] = weights.weights.get(0);
            weightsArray[i * 3 + 1] = weights.weights.get(1);
            weightsArray[i * 3 + 2] = weights.weights.get(2);

        }
        return furthestPoint;
    }

    public void loadNormals() {
        String normalsId = meshData.getChild("polylist").getChildWithAttribute("input", "semantic", "NORMAL")
                .parameters.get("source").substring(1);
        NodeXML normalsData = meshData.getChildWithAttribute("source", "id", normalsId).getChild("float_array");
        int n = normalsData.parameters.get("count") / 3;
        String[] normData = normalsData.data.split(" ");
        for (int i = 0; i < n; i++) {
            float x = normData[i * 3];
            float y = normData[i * 3 + 1];
            float z = normData[i * 3 + 2];
            Vector norm = new Vector(x, y, z, 0f);
            normals.add(new Vector(norm.x, norm.y, norm.z));
        }
    }


    public float[] verticesArray;
    public float[] normalsArray;
    public float[] texturesArray;
    public int[] indicesArray;
    public int[] jointIdsArray;
    public float[] weightsArray;

    public void composeVertecies(){
        NodeXML poly = meshData.getChild("polylist");
        int typeCount = poly.getChildren("input").size();
        String[] indexData = poly.getChild("p").data.split(" ");
        int n = indexData.length / typeCount;
        for(int i = 0; i < n; i++){
            int indexPos = indexData[i * typeCount];
            int indexNorm = indexData[i * typeCount + 1];
            int indexTexCoord = indexData[i * typeCount + 2];
            processVertecies(indexPos, indexNorm, indexTexCoord);
        }
    }

    public void loadPositions() {
        String positionsId = meshData.getChild("vertices").getChild("input").parameters.get("source").substring(1);
        NodeXML positionsData = meshData.getChildWithAttribute("source", "id", positionsId).getChild("float_array");
        int n = positionsData.parameters.get("count") / 3;
        for (int i = 0; i < n; i++) {
            float x = positions[i * 3];
            float y = positions[i * 3 + 1];
            float z = positions[i * 3 + 2];
            Vector position = new Vector(x, y, z);
            vertices.add(new Vertex(vertices.size(), new Vector(position.x, position.y, position.z), vertexWeights.get(vertices.size())));
        }
    }


    public void loadTextures() {
        String texCoordsId = meshData.getChild("polylist").getChildWithAttribute("input", "semantic", "TEXCOORD")
                .parameters.get("source").substring(1);
        NodeXML texCoordsData = meshData.getChildWithAttribute("source", "id", texCoordsId).getChild("float_array");
        int n = texCoordsData.parameters.get("count")) / 2;

        for (int i = 0; i < n; i++) {
            float s = textures[i * 2]);
            float t = textures[i * 2 + 1]);
            textures.add(new Vector(s, t, 0));
        }
    }


}

package main.java.CudaCompute;

import static io.grpc.MethodDescriptor.generateFullMethodName;

/**
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.64.0)",
    comments = "Source: cuda_compute.proto")
@io.grpc.stub.annotations.GrpcGenerated
public final class CudaComputeServiceGrpc {

  private CudaComputeServiceGrpc() {}

  public static final java.lang.String SERVICE_NAME = "cuda_compute.CudaComputeService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<main.java.CudaCompute.CudaCompute.CudaComputeRequest,
      main.java.CudaCompute.CudaCompute.CudaComputeResponse> getCudaComputeMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "CudaCompute",
      requestType = main.java.CudaCompute.CudaCompute.CudaComputeRequest.class,
      responseType = main.java.CudaCompute.CudaCompute.CudaComputeResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<main.java.CudaCompute.CudaCompute.CudaComputeRequest,
      main.java.CudaCompute.CudaCompute.CudaComputeResponse> getCudaComputeMethod() {
    io.grpc.MethodDescriptor<main.java.CudaCompute.CudaCompute.CudaComputeRequest, main.java.CudaCompute.CudaCompute.CudaComputeResponse> getCudaComputeMethod;
    if ((getCudaComputeMethod = CudaComputeServiceGrpc.getCudaComputeMethod) == null) {
      synchronized (CudaComputeServiceGrpc.class) {
        if ((getCudaComputeMethod = CudaComputeServiceGrpc.getCudaComputeMethod) == null) {
          CudaComputeServiceGrpc.getCudaComputeMethod = getCudaComputeMethod =
              io.grpc.MethodDescriptor.<main.java.CudaCompute.CudaCompute.CudaComputeRequest, main.java.CudaCompute.CudaCompute.CudaComputeResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(SERVICE_NAME, "CudaCompute"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  main.java.CudaCompute.CudaCompute.CudaComputeRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  main.java.CudaCompute.CudaCompute.CudaComputeResponse.getDefaultInstance()))
              .setSchemaDescriptor(new CudaComputeServiceMethodDescriptorSupplier("CudaCompute"))
              .build();
        }
      }
    }
    return getCudaComputeMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static CudaComputeServiceStub newStub(io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CudaComputeServiceStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CudaComputeServiceStub>() {
        @java.lang.Override
        public CudaComputeServiceStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CudaComputeServiceStub(channel, callOptions);
        }
      };
    return CudaComputeServiceStub.newStub(factory, channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static CudaComputeServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CudaComputeServiceBlockingStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CudaComputeServiceBlockingStub>() {
        @java.lang.Override
        public CudaComputeServiceBlockingStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CudaComputeServiceBlockingStub(channel, callOptions);
        }
      };
    return CudaComputeServiceBlockingStub.newStub(factory, channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static CudaComputeServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    io.grpc.stub.AbstractStub.StubFactory<CudaComputeServiceFutureStub> factory =
      new io.grpc.stub.AbstractStub.StubFactory<CudaComputeServiceFutureStub>() {
        @java.lang.Override
        public CudaComputeServiceFutureStub newStub(io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
          return new CudaComputeServiceFutureStub(channel, callOptions);
        }
      };
    return CudaComputeServiceFutureStub.newStub(factory, channel);
  }

  /**
   */
  public interface AsyncService {

    /**
     */
    default void cudaCompute(main.java.CudaCompute.CudaCompute.CudaComputeRequest request,
        io.grpc.stub.StreamObserver<main.java.CudaCompute.CudaCompute.CudaComputeResponse> responseObserver) {
      io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall(getCudaComputeMethod(), responseObserver);
    }
  }

  /**
   * Base class for the server implementation of the service CudaComputeService.
   */
  public static abstract class CudaComputeServiceImplBase
      implements io.grpc.BindableService, AsyncService {

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return CudaComputeServiceGrpc.bindService(this);
    }
  }

  /**
   * A stub to allow clients to do asynchronous rpc calls to service CudaComputeService.
   */
  public static final class CudaComputeServiceStub
      extends io.grpc.stub.AbstractAsyncStub<CudaComputeServiceStub> {
    private CudaComputeServiceStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CudaComputeServiceStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CudaComputeServiceStub(channel, callOptions);
    }

    /**
     */
    public void cudaCompute(main.java.CudaCompute.CudaCompute.CudaComputeRequest request,
        io.grpc.stub.StreamObserver<main.java.CudaCompute.CudaCompute.CudaComputeResponse> responseObserver) {
      io.grpc.stub.ClientCalls.asyncUnaryCall(
          getChannel().newCall(getCudaComputeMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * A stub to allow clients to do synchronous rpc calls to service CudaComputeService.
   */
  public static final class CudaComputeServiceBlockingStub
      extends io.grpc.stub.AbstractBlockingStub<CudaComputeServiceBlockingStub> {
    private CudaComputeServiceBlockingStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CudaComputeServiceBlockingStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CudaComputeServiceBlockingStub(channel, callOptions);
    }

    /**
     */
    public main.java.CudaCompute.CudaCompute.CudaComputeResponse cudaCompute(main.java.CudaCompute.CudaCompute.CudaComputeRequest request) {
      return io.grpc.stub.ClientCalls.blockingUnaryCall(
          getChannel(), getCudaComputeMethod(), getCallOptions(), request);
    }
  }

  /**
   * A stub to allow clients to do ListenableFuture-style rpc calls to service CudaComputeService.
   */
  public static final class CudaComputeServiceFutureStub
      extends io.grpc.stub.AbstractFutureStub<CudaComputeServiceFutureStub> {
    private CudaComputeServiceFutureStub(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected CudaComputeServiceFutureStub build(
        io.grpc.Channel channel, io.grpc.CallOptions callOptions) {
      return new CudaComputeServiceFutureStub(channel, callOptions);
    }

    /**
     */
    public com.google.common.util.concurrent.ListenableFuture<main.java.CudaCompute.CudaCompute.CudaComputeResponse> cudaCompute(
        main.java.CudaCompute.CudaCompute.CudaComputeRequest request) {
      return io.grpc.stub.ClientCalls.futureUnaryCall(
          getChannel().newCall(getCudaComputeMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_CUDA_COMPUTE = 0;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final AsyncService serviceImpl;
    private final int methodId;

    MethodHandlers(AsyncService serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_CUDA_COMPUTE:
          serviceImpl.cudaCompute((main.java.CudaCompute.CudaCompute.CudaComputeRequest) request,
              (io.grpc.stub.StreamObserver<main.java.CudaCompute.CudaCompute.CudaComputeResponse>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  public static final io.grpc.ServerServiceDefinition bindService(AsyncService service) {
    return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
        .addMethod(
          getCudaComputeMethod(),
          io.grpc.stub.ServerCalls.asyncUnaryCall(
            new MethodHandlers<
              main.java.CudaCompute.CudaCompute.CudaComputeRequest,
              main.java.CudaCompute.CudaCompute.CudaComputeResponse>(
                service, METHODID_CUDA_COMPUTE)))
        .build();
  }

  private static abstract class CudaComputeServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    CudaComputeServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return main.java.CudaCompute.CudaCompute.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("CudaComputeService");
    }
  }

  private static final class CudaComputeServiceFileDescriptorSupplier
      extends CudaComputeServiceBaseDescriptorSupplier {
    CudaComputeServiceFileDescriptorSupplier() {}
  }

  private static final class CudaComputeServiceMethodDescriptorSupplier
      extends CudaComputeServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final java.lang.String methodName;

    CudaComputeServiceMethodDescriptorSupplier(java.lang.String methodName) {
      this.methodName = methodName;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (CudaComputeServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new CudaComputeServiceFileDescriptorSupplier())
              .addMethod(getCudaComputeMethod())
              .build();
        }
      }
    }
    return result;
  }
}

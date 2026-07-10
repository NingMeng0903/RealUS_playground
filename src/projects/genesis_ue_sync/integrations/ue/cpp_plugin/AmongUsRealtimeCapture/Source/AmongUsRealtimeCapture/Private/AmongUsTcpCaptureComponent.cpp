#include "AmongUsTcpCaptureComponent.h"

#include "Components/SceneCaptureComponent2D.h"
#include "Dom/JsonObject.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HAL/PlatformTime.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "IPAddress.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "Misc/DateTime.h"
#include "Modules/ModuleManager.h"
#include "Policies/CondensedJsonPrintPolicy.h"
#include "RenderingThread.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "TextureResource.h"
#include "UnrealClient.h"

namespace
{
bool AppendUint32BE(TArray<uint8>& Buf, uint32 Value)
{
	Buf.Add(static_cast<uint8>((Value >> 24) & 0xff));
	Buf.Add(static_cast<uint8>((Value >> 16) & 0xff));
	Buf.Add(static_cast<uint8>((Value >> 8) & 0xff));
	Buf.Add(static_cast<uint8>(Value & 0xff));
	return true;
}

bool SendAllBytes(FSocket* Socket, const uint8* Data, int32 Remaining)
{
	while (Remaining > 0)
	{
		int32 Sent = 0;
		if (!Socket->Send(Data, Remaining, Sent) || Sent <= 0)
		{
			return false;
		}
		Data += Sent;
		Remaining -= Sent;
	}
	return true;
}

int64 UnixTimeNs()
{
	const FDateTime Now = FDateTime::UtcNow();
	return Now.ToUnixTimestamp() * 1000000000LL + static_cast<int64>(Now.GetMillisecond()) * 1000000LL;
}
} // namespace

UAmongUsTcpCaptureComponent::UAmongUsTcpCaptureComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	TcpHost = TEXT("127.0.0.1");
	TcpPort = 17355;
	JpegQuality = 85;
	bAutoConnect = true;
	ConnectRetrySeconds = 2.0f;
	FallbackDtNs = 10000000LL;
	TcpSocket = nullptr;
	ExternalFrameIndex = -1;
	ExternalSimTimeNs = -1;
	LastConnectAttemptSeconds = 0.0;
	InternalFrameCounter = 0;
	InternalSimTimeNs = 0;
}

void UAmongUsTcpCaptureComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	DisconnectSocket();
	Super::EndPlay(EndPlayReason);
}

void UAmongUsTcpCaptureComponent::SetExternalSimClock(int64 FrameIndex, int64 SimTimeNs)
{
	ExternalFrameIndex = FrameIndex;
	ExternalSimTimeNs = SimTimeNs;
}

void UAmongUsTcpCaptureComponent::SetSessionId(const FString& InSessionId)
{
	SessionId = InSessionId;
}

void UAmongUsTcpCaptureComponent::DisconnectSocket()
{
	if (TcpSocket == nullptr)
	{
		return;
	}
	ISocketSubsystem* Subsys = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
	if (Subsys != nullptr)
	{
		Subsys->DestroySocket(TcpSocket);
	}
	TcpSocket = nullptr;
}

bool UAmongUsTcpCaptureComponent::EnsureConnected()
{
	if (TcpSocket != nullptr && TcpSocket->GetConnectionState() == ESocketConnectionState::SCS_Connected)
	{
		return true;
	}
	DisconnectSocket();

	const double NowSec = FPlatformTime::Seconds();
	if ((NowSec - LastConnectAttemptSeconds) < static_cast<double>(ConnectRetrySeconds))
	{
		return false;
	}
	LastConnectAttemptSeconds = NowSec;

	ISocketSubsystem* Subsys = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
	if (Subsys == nullptr)
	{
		return false;
	}

	FSocket* NewSock = Subsys->CreateSocket(NAME_Stream, TEXT("AmongUsTcpCapture"), false);
	if (NewSock == nullptr)
	{
		return false;
	}

	NewSock->SetNonBlocking(false);

	TSharedRef<FInternetAddr> Addr = Subsys->CreateInternetAddr();
	bool bIsValid = false;
	Addr->SetIp(*TcpHost, bIsValid);
	if (!bIsValid)
	{
		FIPv4Address Parsed;
		if (FIPv4Address::Parse(TcpHost, Parsed))
		{
			Addr->SetIp(Parsed.Value);
			bIsValid = true;
		}
	}
	if (!bIsValid)
	{
		Subsys->DestroySocket(NewSock);
		return false;
	}

	Addr->SetPort(TcpPort);
	if (!NewSock->Connect(*Addr))
	{
		Subsys->DestroySocket(NewSock);
		return false;
	}

	TcpSocket = NewSock;
	return true;
}

bool UAmongUsTcpCaptureComponent::RenderTargetToJpeg(UTextureRenderTarget2D* RenderTarget, TArray<uint8>& OutBytes,
													 int32& OutW, int32& OutH, bool bFlipU, bool bFlipV)
{
	OutBytes.Reset();
	if (RenderTarget == nullptr)
	{
		return false;
	}

	FlushRenderingCommands();

	FTextureRenderTargetResource* RtResource = RenderTarget->GameThread_GetRenderTargetResource();
	if (RtResource == nullptr)
	{
		return false;
	}

	TArray<FColor> Bitmap;
	RtResource->ReadPixels(Bitmap);

	const int32 Width = RenderTarget->SizeX;
	const int32 Height = RenderTarget->SizeY;
	if (Width <= 0 || Height <= 0 || Bitmap.Num() != Width * Height)
	{
		return false;
	}

	if (bFlipU || bFlipV)
	{
		TArray<FColor> Corrected;
		Corrected.SetNumUninitialized(Bitmap.Num());
		for (int32 Y = 0; Y < Height; ++Y)
		{
			for (int32 X = 0; X < Width; ++X)
			{
				const int32 SrcX = bFlipU ? (Width - 1 - X) : X;
				const int32 SrcY = bFlipV ? (Height - 1 - Y) : Y;
				Corrected[Y * Width + X] = Bitmap[SrcY * Width + SrcX];
			}
		}
		Bitmap = MoveTemp(Corrected);
	}

	TArray<uint8> RawRgba;
	RawRgba.SetNumUninitialized(Bitmap.Num() * 4);
	for (int32 i = 0; i < Bitmap.Num(); ++i)
	{
		const FColor& C = Bitmap[i];
		RawRgba[i * 4 + 0] = C.R;
		RawRgba[i * 4 + 1] = C.G;
		RawRgba[i * 4 + 2] = C.B;
		RawRgba[i * 4 + 3] = C.A;
	}

	IImageWrapperModule& ImageWrapperModule =
		FModuleManager::LoadModuleChecked<IImageWrapperModule>(FName("ImageWrapper"));
	TSharedPtr<IImageWrapper> Wrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::JPEG);
	if (!Wrapper.IsValid())
	{
		return false;
	}
	if (!Wrapper->SetRaw(RawRgba.GetData(), RawRgba.Num(), Width, Height, ERGBFormat::RGBA, 8))
	{
		return false;
	}

	auto Compressed = Wrapper->GetCompressed(static_cast<int32>(JpegQuality));
	OutBytes.Append(Compressed.GetData(), Compressed.Num());
	OutW = Width;
	OutH = Height;
	return OutBytes.Num() > 0;
}

FString UAmongUsTcpCaptureComponent::BuildMetadataJson(USceneCaptureComponent2D* Capture, const FString& CameraName,
													   int32 Width, int32 Height, int64 FrameIndex, int64 SimTimeNs,
													   int64 WallTimeNs, bool bFlipU, bool bFlipV)
{
	TSharedPtr<FJsonObject> Root = MakeShared<FJsonObject>();
	Root->SetNumberField(TEXT("schema_version"), 1);
	Root->SetStringField(TEXT("session_id"), SessionId);
	Root->SetStringField(TEXT("source_id"), TEXT("ue.realtime_capture"));
	Root->SetStringField(TEXT("camera_name"), CameraName);
	Root->SetStringField(TEXT("camera_frame_id"), CameraName);
	Root->SetNumberField(TEXT("frame_index"), static_cast<double>(FrameIndex));
	Root->SetNumberField(TEXT("sim_time_ns"), static_cast<double>(SimTimeNs));
	Root->SetNumberField(TEXT("wall_time_ns"), static_cast<double>(WallTimeNs));
	Root->SetNumberField(TEXT("source_time_ns"), static_cast<double>(WallTimeNs));
	Root->SetStringField(TEXT("encoding"), TEXT("jpeg"));
	Root->SetNumberField(TEXT("width"), Width);
	Root->SetNumberField(TEXT("height"), Height);
	Root->SetBoolField(TEXT("scene_capture_flip_u"), bFlipU);
	Root->SetBoolField(TEXT("scene_capture_flip_v"), bFlipV);

	TSharedPtr<FJsonObject> Intrinsics = MakeShared<FJsonObject>();
	if (Capture != nullptr)
	{
		Intrinsics->SetNumberField(TEXT("fov_degrees"), Capture->FOVAngle);
	}
	Intrinsics->SetNumberField(TEXT("width"), Width);
	Intrinsics->SetNumberField(TEXT("height"), Height);
	Root->SetObjectField(TEXT("intrinsics"), Intrinsics);

	TSharedPtr<FJsonObject> Extrinsics = MakeShared<FJsonObject>();
	if (Capture != nullptr)
	{
		const FVector Loc = Capture->GetComponentLocation();
		const FRotator Rot = Capture->GetComponentRotation();
		TArray<TSharedPtr<FJsonValue>> LocCm;
		LocCm.Add(MakeShared<FJsonValueNumber>(Loc.X));
		LocCm.Add(MakeShared<FJsonValueNumber>(Loc.Y));
		LocCm.Add(MakeShared<FJsonValueNumber>(Loc.Z));
		TArray<TSharedPtr<FJsonValue>> RotDeg;
		RotDeg.Add(MakeShared<FJsonValueNumber>(Rot.Pitch));
		RotDeg.Add(MakeShared<FJsonValueNumber>(Rot.Yaw));
		RotDeg.Add(MakeShared<FJsonValueNumber>(Rot.Roll));
		Extrinsics->SetArrayField(TEXT("ue_location_cm"), LocCm);
		Extrinsics->SetArrayField(TEXT("ue_rotation_deg"), RotDeg);
	}
	Root->SetObjectField(TEXT("extrinsics"), Extrinsics);

	FString Out;
	TSharedRef<TJsonWriter<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>> Writer =
		TJsonWriterFactory<TCHAR, TCondensedJsonPrintPolicy<TCHAR>>::Create(&Out);
	FJsonSerializer::Serialize(Root.ToSharedRef(), Writer);
	return Out;
}

bool UAmongUsTcpCaptureComponent::SendFramedPacket(const FString& MetaJson, const TArray<uint8>& ImageBytes)
{
	if (TcpSocket == nullptr)
	{
		return false;
	}

	FTCHARToUTF8 MetaUtf8(*MetaJson);
	TArray<uint8> Packet;
	const uint32 MetaLen = static_cast<uint32>(MetaUtf8.Length());
	if (!AppendUint32BE(Packet, MetaLen))
	{
		return false;
	}
	Packet.Append(reinterpret_cast<const uint8*>(MetaUtf8.Get()), MetaUtf8.Length());

	const uint32 ImgLen = static_cast<uint32>(ImageBytes.Num());
	if (!AppendUint32BE(Packet, ImgLen))
	{
		return false;
	}
	Packet.Append(ImageBytes);

	return SendAllBytes(TcpSocket, Packet.GetData(), Packet.Num());
}

void UAmongUsTcpCaptureComponent::TickComponent(float DeltaTime, ELevelTick TickType,
												FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (SceneCaptures.Num() <= 0)
	{
		return;
	}

	if (TcpSocket == nullptr || TcpSocket->GetConnectionState() != ESocketConnectionState::SCS_Connected)
	{
		if (!bAutoConnect)
		{
			return;
		}
		if (!EnsureConnected())
		{
			return;
		}
	}

	const bool bUseExternalClock = ExternalFrameIndex >= 0 && ExternalSimTimeNs >= 0;
	const int64 WallNs = UnixTimeNs();
	const int64 FrameForMeta = bUseExternalClock ? ExternalFrameIndex : InternalFrameCounter;
	const int64 SimNsForMeta = bUseExternalClock ? ExternalSimTimeNs : InternalSimTimeNs;

	const int32 NumCaps = SceneCaptures.Num();
	for (int32 Idx = 0; Idx < NumCaps; ++Idx)
	{
		USceneCaptureComponent2D* Cap = SceneCaptures[Idx];
		if (Cap == nullptr || Cap->TextureTarget == nullptr)
		{
			continue;
		}

		Cap->CaptureScene();

		const bool bFlipU = CameraFlipU.IsValidIndex(Idx) ? CameraFlipU[Idx] : false;
		const bool bFlipV = CameraFlipV.IsValidIndex(Idx) ? CameraFlipV[Idx] : false;

		TArray<uint8> Jpeg;
		int32 W = 0;
		int32 H = 0;
		if (!RenderTargetToJpeg(Cap->TextureTarget, Jpeg, W, H, bFlipU, bFlipV))
		{
			continue;
		}

		FString CamName =
			CameraNames.IsValidIndex(Idx) ? CameraNames[Idx] : FString::Printf(TEXT("camera_%d"), Idx);
		const FString Meta =
			BuildMetadataJson(Cap, CamName, W, H, FrameForMeta, SimNsForMeta, WallNs, bFlipU, bFlipV);
		if (!SendFramedPacket(Meta, Jpeg))
		{
			DisconnectSocket();
			return;
		}
	}

	if (!bUseExternalClock)
	{
		InternalFrameCounter++;
		InternalSimTimeNs += FallbackDtNs;
	}
}
